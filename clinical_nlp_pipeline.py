#!/usr/bin/env python3
"""Offline-first clinical concept extraction pipeline.

The module deliberately has no mandatory network/model dependency.  It combines
high precision medication/lab/context rules with configurable ICD/RxNorm
indexes, and asks an OpenAI-compatible *localhost* LLM for semantic spans.
All public spans are aligned back to the
original string before serialization.
"""
from __future__ import annotations

import argparse, csv, json, logging, re, sys, unicodedata, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ontology_linking import DerivedOntologyIndex, ScoredCandidate

try:
    from rapidfuzz import fuzz, process  # optional
except Exception:  # pragma: no cover
    fuzz = None; process = None

ENTITY_TYPES = {"TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC"}
ASSERTION_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
ALLOWED_ASSERTIONS = ("isNegated", "isFamily", "isHistorical")

# No clinical concepts are embedded in code. ICD/RxNorm and optional lab or
# symptom dictionaries are loaded from versioned files supplied by the operator.

def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()

def loose(s: str) -> str:
    return re.sub(r"[^\w%]+", "", norm(s), flags=re.UNICODE)

@dataclass
class Entity:
    text: str; typ: str; start: int; end: int
    assertions: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    confidence: float = 0.0; source: str = "rule"
    def public(self) -> dict:
        d = {"text": self.text, "type": self.typ, "position": [self.start, self.end]}
        if self.typ in ASSERTION_TYPES: d["assertions"] = [x for x in ALLOWED_ASSERTIONS if x in self.assertions]
        if self.typ in {"CHẨN_ĐOÁN", "THUỐC"}: d["candidates"] = [str(x) for x in self.candidates]
        return d


@dataclass(frozen=True)
class LineView:
    """A raw line with offsets into the original document."""

    line_id: str
    start: int
    end: int
    text: str
    heading_end: int | None = None
    section_kind: str = "neutral"


@dataclass(frozen=True)
class SectionView:
    start: int
    end: int
    kind: str
    current: bool = False
    historical: bool = False


CURRENT_SECTION_PATTERNS = (
    "bệnh sử hiện tại",
    "tiền sử bệnh hiện tại",
    "triệu chứng hiện tại",
    "các triệu chứng hiện tại",
    "lý do nhập viện",
    "tình trạng lúc vào viện",
    "tình trạng ngay trước khi nhập viện",
)
HISTORY_SECTION_PATTERNS = (
    "tiền sử bệnh",
    "thuốc trước khi nhập viện",
    "các yếu tố nguy cơ liên quan",
    "tiền sử phẫu thuật",
    "bệnh đã điều trị trước đây",
    "lịch sử bệnh",
)
SECTION_PREFIXES = tuple(dict.fromkeys(CURRENT_SECTION_PATTERNS + HISTORY_SECTION_PATTERNS + (
    "kết quả khám lâm sàng",
    "kết quả thăm khám lâm sàng",
    "kết quả xét nghiệm",
    "kết quả chẩn đoán hình ảnh",
    "kết quả chụp ảnh/kỹ thuật chẩn đoán hình ảnh",
    "các kết quả chẩn đoán khác",
)))

METADATA_LABEL_RE = re.compile(
    r"^\s*(?:vị trí|vùng|thời gian|mức độ|đặc điểm|yếu tố làm nặng thêm|yếu tố giảm nhẹ|"
    r"yếu tố liên quan|tần suất|khởi phát|đường dùng)\s*:", re.IGNORECASE,
)
DRUG_ACTION_PREFIX_RE = re.compile(
    r"^\s*(?:bắt đầu\s+)?(?:đang\s+|đã\s+)?(?:dùng|uống|tiêm|truyền)\s+",
    re.IGNORECASE,
)
PROCEDURE_CUES = (
    "cắt bỏ", "phẫu thuật", "thủ thuật", "chọc hút", "nạo hạch", "đặt stent",
    "stent graft", "reduced from", "đã khám tổn thương", "mổ ", "nội soi",
)


def _strip_heading_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:[-*•]\s*)?(?:\d+(?:\.\d+)*[.)]?\s*)?", "", text)


def _generic_heading_end(text: str) -> int | None:
    stripped = text.strip()
    if not stripped:
        return None
    if re.match(r"^\s*(?:[-*•]\s*)?\d+(?:\.\d+)*[.)]?\s+", text):
        return len(text.rstrip())
    if stripped.endswith(":") and len(stripped) <= 100:
        return len(text.rstrip())
    return None


def _classify_section(text: str) -> tuple[str, bool, bool, int | None]:
    """Classify a heading prefix; current is deliberately checked first."""
    body = _strip_heading_prefix(text)
    folded = norm(body).rstrip(":")
    for phrase in CURRENT_SECTION_PATTERNS:
        if folded.startswith(phrase):
            end = len(text) - len(body) + len(phrase)
            return "current", True, False, end
    for phrase in HISTORY_SECTION_PATTERNS:
        if folded.startswith(phrase):
            end = len(text) - len(body) + len(phrase)
            return "historical", False, True, end
    for phrase in SECTION_PREFIXES:
        if folded.startswith(phrase):
            end = len(text) - len(body) + len(phrase)
            return "neutral", False, False, end
    return "neutral", False, False, _generic_heading_end(text)


@dataclass
class DocumentView:
    raw: str
    lines: list[LineView]
    sections: list[SectionView]

    @classmethod
    def build(cls, raw: str) -> "DocumentView":
        lines: list[LineView] = []
        cursor = 0
        chunks = raw.splitlines(keepends=True) or [""]
        for index, chunk in enumerate(chunks, 1):
            text = chunk.rstrip("\r\n")
            kind, current, historical, heading_len = _classify_section(text)
            lines.append(LineView(
                line_id=f"L{index:03d}",
                start=cursor,
                end=cursor + len(text),
                text=text,
                heading_end=(cursor + heading_len if heading_len is not None else None),
                section_kind=kind,
            ))
            cursor += len(chunk)

        starts: list[tuple[int, str, bool, bool]] = [
            (line.start, line.section_kind, line.section_kind == "current", line.section_kind == "historical")
            for line in lines if line.section_kind != "neutral" or line.heading_end is not None
        ]
        sections: list[SectionView] = []
        for index, (start, kind, current, historical) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(raw)
            sections.append(SectionView(start, end, kind, current, historical))
        return cls(raw, lines, sections)

    def line_for(self, pos: int) -> LineView:
        for line in self.lines:
            if line.start <= pos <= line.end:
                return line
        return self.lines[-1]

    def section_for(self, pos: int) -> SectionView:
        selected = SectionView(0, len(self.raw), "neutral")
        for section in self.sections:
            if section.start <= pos < section.end:
                selected = section
        return selected

    def is_heading_entity(self, start: int, end: int) -> bool:
        line = self.line_for(start)
        return line.heading_end is not None and end <= line.heading_end

    def is_metadata_entity(self, start: int, end: int) -> bool:
        return bool(METADATA_LABEL_RE.match(self.raw[start:end]))

class Ontology:
    def __init__(self, path: str | None, kind: str = "ontology"):
        self.kind, self.rows, self.index = kind.casefold(), [], None
        if path:
            p = Path(path)
            if not p.exists(): raise FileNotFoundError(path)
            alias_path = p.parent.parent / "derived" / "aliases.jsonl"
            manifest_path = p.parent.parent / "derived" / "manifest.json"
            alias_table = alias_path if alias_path.exists() else None
            manifest = manifest_path if manifest_path.exists() and p.parent.name == self.kind else None
            if p.suffix.lower() in {".json", ".jsonl"}:
                self.index = DerivedOntologyIndex(p, alias_table=alias_table, manifest=manifest, kind=self.kind)
                for concept in self.index.concepts:
                    for name in concept.aliases:
                        self.rows.append((name, [concept.code]))
                for record in self.index.aliases:
                    if record.confidence >= 0.80 and record.allow_public:
                        self.rows.append((record.alias, [record.code]))
            else:
                self._load(Path(path))
        self.alias_to_codes: dict[str, list[str]] = {}
        for n, codes in self.rows:
            bucket = self.alias_to_codes.setdefault(norm(n), [])
            for code in codes:
                if code not in bucket: bucket.append(code)
        self.rows_by_length = sorted(self.rows, key=lambda x: -len(x[0]))
        self.rows_by_first: dict[str, list[tuple[str, list[str]]]] = {}
        for row in self.rows:
            first = norm(row[0]).split(" ", 1)[0]
            self.rows_by_first.setdefault(first, []).append(row)
        if path and not self.rows:
            raise ValueError(f"ontology index is empty: {path}")

    def _load(self, p: Path):
        try:
            if p.suffix.lower() in {".json", ".jsonl"}:
                data = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                if isinstance(data, dict): data = data.get("concepts", [])
                for r in data:
                    code = r.get("code") or r.get("rxcui") or r.get("RXCUI") or r.get("id")
                    label = r.get("label") or r.get("name") or r.get("STR") or r.get("preferred_name")
                    aliases = r.get("aliases", []) or []
                    if code and label:
                        for n in [label, *aliases]: self.rows.append((str(n), [str(code)]))
            else:
                with p.open(encoding="utf-8", newline="") as f:
                    for r in csv.DictReader(f):
                        code = r.get("code") or r.get("rxcui") or r.get("RXCUI") or r.get("id"); label = r.get("label") or r.get("name") or r.get("STR")
                        if code and label: self.rows.append((label, [str(code)]))
        except Exception as e: raise ValueError(f"ontology load failed {p}: {e}") from e

    def lookup_scored(self, mention: str, k: int | None = None) -> list[ScoredCandidate]:
        if self.index:
            return self.index.lookup_scored(mention, kind=self.kind, max_k=k)
        return []

    def lookup(self, mention: str, k: int | None = None) -> list[str]:
        if self.index:
            return self.index.lookup(mention, kind=self.kind, max_k=k)
        cap = k or 3
        q = norm(mention); exact = self.alias_to_codes.get(q)
        if exact: return exact[:cap]
        q2 = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|µg|ml|%|meq)\b", "", q)
        q2 = re.sub(r"\b(?:po|oral|iv|im|sc|sl|daily|bid|tid|qid|qam|qhs|prn|q\d+h|xl|extended release)\b", "", q2).strip()
        core_exact = self.alias_to_codes.get(q2)
        if core_exact:
            return core_exact[:cap]
        scored: list[tuple[float, str]] = []
        tokens = set(re.findall(r"[\wÀ-ỹ]+", q2))
        pool = []
        for token in tokens: pool.extend(self.rows_by_first.get(token, []))
        for n, codes in pool or self.rows:
            name = norm(n)
            if fuzz:
                a = max(fuzz.ratio(q2, name), fuzz.token_set_ratio(q2, name)) / 100
            else:
                a = 1.0 if q2 in name or name in q2 else 0.0
            if a >= .52: scored.append((a, codes[0]))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        ordered: list[str] = []
        for _, code in ranked:
            if code not in ordered:
                ordered.append(code)
        if not ranked:
            return []
        best = ranked[0][0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if best >= .92 and best - second >= .08:
            dynamic_k = 1
        elif best >= .75 and best - second >= .04:
            dynamic_k = 2
        else:
            dynamic_k = 3
        return ordered[:min(cap, dynamic_k)]

def load_terms(path: str | None) -> set[str]:
    """Load one term per line or JSON/JSONL records with a text/label field."""
    if not path: return set()
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    if p.suffix.lower() == ".json":
        value = json.loads(p.read_text(encoding="utf-8")); value = value.get("terms", value) if isinstance(value, dict) else value
        return {str(x.get("text") or x.get("label") or x.get("name")) for x in value if isinstance(x, dict)} | {str(x) for x in value if isinstance(x, str)}
    if p.suffix.lower() == ".jsonl":
        return {str((json.loads(x).get("text") or json.loads(x).get("label") or json.loads(x).get("name"))) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
    return {x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")}
def sentence_window(raw: str, start: int, end: int) -> tuple[int, int, str]:
    boundaries = ".!?;\n"
    before = [raw.rfind(char, 0, start) for char in boundaries]
    a = max(before, default=-1) + 1
    after = [raw.find(char, end) for char in boundaries]
    b0 = [x for x in after if x >= 0]
    b = min(b0) if b0 else len(raw)
    return a, b, raw[a:b]

def section_historical(raw: str, pos: int) -> bool:
    section = DocumentView.build(raw).section_for(pos)
    return section.historical and not section.current


NEGATION_CUES = (
    "không có", "không ghi nhận", "không phát hiện", "không thấy",
    "không bằng chứng", "phủ nhận", "chưa", "âm tính", "không",
)
HISTORY_CUES = (
    "tiền sử", "đã từng", "trước đây", "đã điều trị", "ngừng",
    "ngừng uống", "trước khi nhập viện", "đã dùng", "đang dùng trước",
)
FAMILY_CUES = ("bố", "cha", "mẹ", "anh chị em", "người nhà", "gia đình", "họ hàng")


def _cue_before_entity(text: str, cues: tuple[str, ...]) -> bool:
    folded = norm(text)
    return any(re.search(r"(?:^|[,:;])\s*" + re.escape(cue) + r"\s*$", folded) for cue in cues)


def _has_negation_scope(raw: str, e: Entity, view: DocumentView) -> bool:
    start, _, sentence = sentence_window(raw, e.start, e.end)
    local = norm(raw[max(start, e.start - 30):min(len(raw), e.end + 10)])
    if re.search(r"(?:cà phê|coffee)\s+không\s+caffeine", local):
        return False
    before = raw[start:e.start]
    if _cue_before_entity(before, NEGATION_CUES):
        return True
    # A coordinated list inherits a preceding cue, but only inside this sentence.
    cue_positions = [before.casefold().rfind(cue) for cue in NEGATION_CUES]
    cue_position = max(cue_positions, default=-1)
    if cue_position >= 0:
        between = norm(before[cue_position:])
        if len(between) <= 90 and not re.search(r"\b(?:bắt đầu|dùng|điều trị|cải thiện|đáp ứng)\b", between):
            return True
    # “cà phê không caffeine” is a lexical decaffeinated modifier, not a
    # negated caffeine clinical concept.
    return False


def _has_family_scope(raw: str, e: Entity) -> bool:
    start, end, sentence = sentence_window(raw, e.start, e.end)
    before = norm(raw[start:e.start])
    after = norm(raw[e.end:end])
    family_before = any(cue in before for cue in FAMILY_CUES)
    family_after = any(re.search(r"\b(?:của|ở)\s+" + re.escape(cue), after) for cue in FAMILY_CUES)
    if not (family_before or family_after):
        return False
    relation_words = ("tiền sử", "bị", "mắc", "có", "đau", "triệu chứng", "bệnh", "xuất hiện", "phủ nhận", "không")
    return any(word in before or word in after for word in relation_words)


def _has_local_history(raw: str, e: Entity) -> bool:
    start, _, _ = sentence_window(raw, e.start, e.end)
    before = norm(raw[start:e.start])
    return _cue_before_entity(before, HISTORY_CUES) or any(cue in before for cue in HISTORY_CUES)


def assertions(raw: str, e: Entity, view: DocumentView | None = None) -> list[str]:
    if e.typ not in ASSERTION_TYPES:
        return []
    view = view or DocumentView.build(raw)
    out: list[str] = []
    if _has_negation_scope(raw, e, view):
        out.append("isNegated")
    if _has_family_scope(raw, e):
        out.append("isFamily")
    section = view.section_for(e.start)
    if not section.current and (section.historical or _has_local_history(raw, e)):
        out.append("isHistorical")
    return [name for name in ALLOWED_ASSERTIONS if name in out]

def add_match(out: list[Entity], raw: str, start: int, end: int, typ: str, conf=.7, cand=None, source="rule"):
    if start < 0 or end <= start: return
    text = raw[start:end]
    if not text.strip() or typ not in ENTITY_TYPES: return
    out.append(Entity(text, typ, start, end, candidates=list(cand or []), confidence=conf, source=source))

def find_all(raw: str, phrase: str) -> Iterable[tuple[int, int]]:
    # tolerant whitespace and hyphen variants while returning raw occurrence
    parts = [re.escape(x) for x in re.split(r"[\s-]+", phrase.strip()) if x]
    if not parts: return []
    pat = r"(?i)(?<!\w)" + r"[\s-]+".join(parts) + r"(?!\w)"
    return ((m.start(), m.end()) for m in re.finditer(pat, raw))


RESULT_WORDS = (
    "âm tính", "dương tính", "bình thường", "bất thường", "thấp", "cao",
    "không ghi nhận", "không phát hiện", "không thấy", "ghi nhận", "cho thấy",
)


def _result_after(raw: str, end: int) -> tuple[int, int] | None:
    tail = raw[end:min(len(raw), end + 120)]
    patterns = (
        r"\s*(?::|=|là|cho thấy|ghi nhận|tăng(?:\s+nhẹ)?\s+lên|giảm(?:\s+nhẹ)?\s+xuống)\s*([^\n.;]{1,90})",
        r"\s*((?:không\s+)?(?:ghi nhận|phát hiện|thấy)[^\n.;]{0,70})",
        r"\s*((?:âm tính|dương tính|bình thường|bất thường|thấp|cao))",
        r"\s*([+-]?\d+(?:[.,]\d+)?(?:\s*(?:mg/dl|g/dl|%|mmol/l|umol/l|ng/ml))?)",
    )
    for pattern in patterns:
        match = re.match(pattern, tail, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).rstrip(" ,:;-")
        if value:
            start = end + match.start(1)
            return start, start + len(value)
    return None

def detect(raw: str, icd: Ontology, rx: Ontology, lab_aliases: set[str] | None = None, symptom_aliases: set[str] | None = None) -> list[Entity]:
    out: list[Entity] = []
    lab_aliases = lab_aliases or set()
    symptom_aliases = symptom_aliases or set()
    raw_search = norm(raw)
    generic_leading = {"other", "unspecified", "without", "with", "due", "and", "of", "the", "a", "an", "for", "in"}
    # Ontology aliases are the sole deterministic source of diagnoses.
    for phrase, codes in icd.rows_by_length:
        if len(phrase) < 3: continue
        first = norm(phrase).split(" ", 1)[0]
        if first in generic_leading or (len(first) > 2 and first not in raw_search): continue
        for a, b in find_all(raw, phrase): add_match(out, raw, a, b, "CHẨN_ĐOÁN", .78, codes)
    # Drug aliases and medication windows.
    active_first = set(re.findall(r"[\wÀ-ỹ]+", raw_search))
    drug_names = [n for first in active_first for n, _ in rx.rows_by_first.get(first, [])]
    attrs = r"(?:\s+\(?[\d.,]+\s*(?:mg|g|mcg|µg|ml|meq|%)\)?|\s+(?:po|oral|iv|im|sc|sl|daily|bid|tid|qid|qam|qhs|prn|q\d+h)\b|\s+xl\b|\s+extended release\b){0,8}"
    for name in drug_names:
        first = norm(name).split(" ", 1)[0]
        if len(first) > 2 and first not in raw_search: continue
        for m in re.finditer(r"(?i)(?<!\w)" + r"[\s-]+".join(re.escape(x) for x in re.split(r"\s+", name)) + r"(?!\w)" + attrs, raw):
            a, b = m.start(), m.end()
            local = norm(raw[max(0, a - 30):min(len(raw), b + 20)])
            if re.search(r"(?:cà phê|coffee)\s+không\s+caffeine", local):
                continue
            trimmed = re.sub(r"[\s,;:.-]+$", "", raw[a:b])
            b = a + len(trimmed)
            add_match(out, raw, a, b, "THUỐC", .9, rx.lookup(raw[a:b]))
    # Generic medication line fallback: a likely name plus route/dose or known med cue.
    for m in re.finditer(r"(?im)(?<!\w)([A-Za-z][A-Za-z0-9+/'-]{2,}(?:\s+[A-Za-z][A-Za-z0-9+/'-]{1,}){0,3})(?=\s+(?:\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ml|meq)|po|oral|iv|daily|bid|qid|prn)\b)", raw):
        if norm(m.group(1)) not in {norm(x.text) for x in out if x.typ == "THUỐC"}:
            add_match(out, raw, m.start(), m.end(), "THUỐC", .62, rx.lookup(m.group(1)))
    # Tests and result values: only numbers/words in lab context.
    lab_names = sorted(lab_aliases, key=len, reverse=True)
    for name in lab_names:
        for a, b in find_all(raw, name):
            add_match(out, raw, a, b, "TÊN_XÉT_NGHIỆM", .8)
            value_span = _result_after(raw, b)
            if value_span:
                add_match(out, raw, value_span[0], value_span[1], "KẾT_QUẢ_XÉT_NGHIỆM", .82)
    # Generic lab marker:value (avoid drug dose because nearby medication wins).
    for m in re.finditer(r"(?i)\b([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9%_ /-]{1,35})\s*[:=]\s*([+-]?\d+(?:[.,]\d+)?(?:\s*[A-Za-z%/µ]+)?|âm tính|dương tính|bình thường|bất thường)", raw):
        key = norm(m.group(1));
        if key in {norm(x) for x in lab_aliases}:
            add_match(out, raw, m.start(1), m.end(1), "TÊN_XÉT_NGHIỆM", .75)
            add_match(out, raw, m.start(2), m.end(2), "KẾT_QUẢ_XÉT_NGHIỆM", .75)
    # Symptoms and diagnoses expressed only in clinical prose.
    for phrase in sorted(symptom_aliases, key=len, reverse=True):
        for a, b in find_all(raw, phrase): add_match(out, raw, a, b, "TRIỆU_CHỨNG", .7)
    return out

def _looks_like_result(text: str) -> bool:
    value = norm(text)
    return any(word in value for word in RESULT_WORDS) or bool(re.search(r"\b(?:nhịp|ngoại tâm thu|tăng|giảm)\b", value))


def _looks_like_procedure(text: str) -> bool:
    folded = norm(text)
    return any(cue in folded for cue in PROCEDURE_CUES)


def _trim_entity_context(raw: str, item: Entity) -> Entity | None:
    """Remove action/metadata wrappers while preserving raw offsets."""
    text = raw[item.start:item.end]
    if item.typ == "THUỐC":
        if norm(text) == "reduced from" or _looks_like_procedure(text):
            return None
        match = DRUG_ACTION_PREFIX_RE.match(text)
        if match:
            item.start += match.end()
            item.text = raw[item.start:item.end].strip()
            item.start = item.end - len(item.text)
            item.end = item.start + len(item.text)
        if not item.text or _looks_like_procedure(item.text):
            return None
    elif item.typ == "TRIỆU_CHỨNG":
        match = METADATA_LABEL_RE.match(text)
        if match:
            item.start += match.end()
            while item.start < item.end and raw[item.start].isspace():
                item.start += 1
            item.text = raw[item.start:item.end].strip()
            item.start = item.end - len(item.text)
            item.end = item.start + len(item.text)
            if not item.text:
                return None
    return item


def _choose_same_span(raw: str, group: list[Entity]) -> Entity:
    """Adjudicate detector disagreement for one raw span."""
    types = {item.typ for item in group}
    if {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"} <= types:
        wanted = "KẾT_QUẢ_XÉT_NGHIỆM" if _looks_like_result(raw[group[0].start:group[0].end]) else "TÊN_XÉT_NGHIỆM"
        candidates = [item for item in group if item.typ == wanted]
    else:
        rule_candidates = [item for item in group if item.source != "llm"]
        candidates = rule_candidates or group
    selected = max(candidates, key=lambda item: (item.confidence, item.end - item.start))
    selected.candidates = list(dict.fromkeys(code for item in group for code in item.candidates))
    return selected


def _trim_result_prefix(raw: str, result: Entity, names: list[Entity]) -> None:
    """If an LLM returns ``test name + result`` as one result, keep result text only."""
    for name in names:
        if name.start != result.start or name.end >= result.end:
            continue
        if norm(raw[result.start:name.end]) != norm(name.text):
            continue
        start = name.end
        while start < result.end and raw[start].isspace():
            start += 1
        if start < result.end:
            result.start = start
            result.text = raw[result.start:result.end]
            return


def resolve(raw: str, items: list[Entity]) -> list[Entity]:
    # First remove heading spans and merge detector candidates by exact span/type.
    view = DocumentView.build(raw)
    cleaned: list[Entity] = []
    for item in items:
        if item.start < 0 or item.end <= item.start or item.end > len(raw):
            continue
        if view.is_heading_entity(item.start, item.end):
            continue
        item = _trim_entity_context(raw, item)
        if item is None or item.start < 0 or item.end <= item.start:
            continue
        if view.is_heading_entity(item.start, item.end) or view.is_metadata_entity(item.start, item.end):
            continue
        item.text = raw[item.start:item.end]
        cleaned.append(item)
    items = cleaned
    by_exact: dict[tuple[int, int, str], list[Entity]] = {}
    for e in items:
        by_exact.setdefault((e.start, e.end, e.typ), []).append(e)
    merged = [_choose_same_span(raw, group) for group in by_exact.values()]
    by_span: dict[tuple[int, int], list[Entity]] = {}
    for e in merged:
        by_span.setdefault((e.start, e.end), []).append(e)
    vals = [_choose_same_span(raw, group) for group in by_span.values()]
    names = [e for e in vals if e.typ == "TÊN_XÉT_NGHIỆM"]
    for e in vals:
        if e.typ == "KẾT_QUẢ_XÉT_NGHIỆM":
            _trim_result_prefix(raw, e, names)
    keep = []
    for e in sorted(vals, key=lambda x: (x.start, -(x.end-x.start), -x.confidence)):
        blocked = False
        for q in keep:
            if q.start <= e.start and e.end <= q.end and (q.end-q.start) > (e.end-e.start) and q.typ == e.typ:
                blocked = True; break
        if not blocked: keep.append(e)
    for e in keep:
        e.assertions = assertions(raw, e, view)
        if e.typ in {"CHẨN_ĐOÁN", "THUỐC"}: e.candidates = list(dict.fromkeys(e.candidates))
    return sorted(keep, key=lambda x: (x.start, x.end, x.typ))

def _json_array_from_response(data: dict[str, Any]) -> list[dict]:
    """Extract the first valid JSON array from an OpenAI-compatible response.

    Qwen3/llama.cpp may expose the visible answer in ``content`` or, when
    thinking is enabled, in ``reasoning_content``.  Some server builds also
    return markdown fences around the JSON.  Do not assume either one field
    or that the first/last square bracket delimit the whole answer.
    """
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("LLM response does not contain choices")
    message = choices[0].get("message") or {}
    candidates: list[str] = []
    for key in ("content", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    # A few OpenAI-compatible servers use a top-level text field.
    value = choices[0].get("text")
    if isinstance(value, str) and value.strip():
        candidates.append(value)
    decoder = json.JSONDecoder()
    for text in candidates:
        # Try every '[' because prose or a markdown fence may precede JSON.
        for match in re.finditer(r"\[", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                return value
    # llama.cpp can hit the output-token limit after emitting several complete
    # objects but before writing the closing ']' (common when a model repeats
    # entities). Recover those complete objects instead of discarding the
    # entire document. The incomplete final object is intentionally ignored.
    recovered: list[dict] = []
    for text in candidates:
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and ("text" in value or "type" in value):
                recovered.append(value)
    if recovered:
        logging.warning("LLM JSON was truncated; recovered %d complete items", len(recovered))
        return recovered
    preview = " | ".join(x[-240:] for x in candidates)[:600]
    raise RuntimeError(f"LLM response does not contain a JSON array; response tail={preview!r}")


def llm_extract(raw: str, endpoint: str | None, model: str | None, timeout: int = 180) -> list[dict]:
    if not endpoint: return []
    logging.info("LLM extraction request: endpoint=%s model=%s chars=%d", endpoint, model or "local", len(raw))
    view = DocumentView.build(raw)
    numbered = "\n".join(f"{line.line_id}: {line.text}" for line in view.lines)
    prompt = ("Extract Vietnamese clinical entities from the numbered INPUT. Return JSON array only. "
              "Each item must be {line_id,text,type}; copy text exactly and never invent text. "
              "Do not merge different occurrences. Do not return assertions or positions. "
              "For medication entities, preserve the complete contiguous medication mention exactly as written, "
              "including the core drug name, strength, route, frequency, and sig when present. "
              "For downstream RxNorm linking, reason over the core ingredient and strength: treat "
              "po/bid/tid/qid/daily/prn and similar administration instructions as noise, and allow "
              "a base ingredient to match an RxNorm alias containing a salt, ester, active-moiety "
              "variant, combination component, or derivative. Prefer aliases and semantic proximity "
              "over exact whole-string matching; never replace the copied raw span with a canonical name. "
              "Do not return section headings such as khám lâm sàng, chẩn đoán hình ảnh, or kết quả xét nghiệm. "
              "A test name and its result are different entities: the name is TÊN_XÉT_NGHIỆM; "
              "the measured/textual finding is KẾT_QUẢ_XÉT_NGHIỆM. "
              "Types: TRIỆU_CHỨNG,TÊN_XÉT_NGHIỆM,KẾT_QUẢ_XÉT_NGHIỆM,CHẨN_ĐOÁN,THUỐC. "
              "INPUT:\n" + numbered + "\n/no_think")
    payload = {"model": model or "local", "messages":[{"role":"system","content":"You are a clinical NLP extraction module. Return JSON only. Do not think step by step. Use one line_id per occurrence. /no_think"},{"role":"user","content":prompt}], "temperature":0, "max_tokens":2800}
    try:
        base = endpoint.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, body, {"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r: data = json.loads(r.read().decode())
        value = _json_array_from_response(data)
        logging.info("LLM extraction response: %d items", len(value))
        return value
    except Exception as e:
        raise RuntimeError(f"localhost LLM request/parse failed: {e}") from e

def align_llm(raw: str, items: list[dict]) -> list[Entity]:
    view = DocumentView.build(raw)
    out: list[Entity] = []
    for x in items:
        text, typ = str(x.get("text", "")), x.get("type")
        if typ not in ENTITY_TYPES or not text: continue
        line_id = x.get("line_id")
        search_raw = raw
        offset = 0
        if isinstance(line_id, str):
            line = next((line for line in view.lines if line.line_id == line_id), None)
            if line is None:
                continue
            search_raw = raw[line.start:line.end]
            offset = line.start
        occurrences = list(find_all(search_raw, text))
        # line_id makes repeated occurrences explicit. For legacy responses
        # without line_id, use only the first occurrence to avoid multiplying
        # one ambiguous response across the document.
        if not line_id:
            occurrences = occurrences[:1]
        for a, b in occurrences:
            a += offset; b += offset
            if view.is_heading_entity(a, b):
                continue
            out.append(Entity(raw[a:b], typ, a, b, confidence=.55, source="llm"))
    return out


def _link_index(index: Ontology | DerivedOntologyIndex, mention: str) -> list[ScoredCandidate]:
    max_k = 3 if getattr(index, "kind", "") == "icd" else 2
    scored = index.lookup_scored(mention, k=max_k) if isinstance(index, Ontology) else index.lookup_scored(mention, max_k=max_k)
    if scored:
        return scored
    # Use a curated head when the LLM span contains modifiers or a second
    # clause. This never creates aliases; it only reuses approved rows.
    aliases = getattr(getattr(index, "index", index), "aliases", [])
    for record in sorted(aliases, key=lambda item: len(item.alias), reverse=True):
        if record.confidence < 0.80 or not record.allow_public:
            continue
        if norm(record.alias) in norm(mention):
            scored = index.lookup_scored(record.alias, k=max_k) if isinstance(index, Ontology) else index.lookup_scored(record.alias, max_k=max_k)
            if scored:
                return scored
    return []


def link_entities(
    entities: list[Entity],
    icd_index: Ontology | DerivedOntologyIndex,
    rxnorm_index: Ontology | DerivedOntologyIndex,
    audit: list[dict[str, Any]] | None = None,
) -> list[Entity]:
    """Link only final diagnosis/drug entities after span/type adjudication."""
    for entity in entities:
        entity.candidates = []
        if entity.typ not in {"CHẨN_ĐOÁN", "THUỐC"}:
            continue
        index = icd_index if entity.typ == "CHẨN_ĐOÁN" else rxnorm_index
        scored = _link_index(index, entity.text)
        entity.candidates = [item.code for item in scored]
        if audit is not None:
            audit.append({
                "text": entity.text,
                "type": entity.typ,
                "codes": entity.candidates,
                "match_mode": scored[0].match_mode if scored else "no_hit",
                "score": scored[0].score if scored else 0.0,
                "margin": scored[0].margin if scored else 0.0,
                "evidence": scored[0].evidence if scored else {},
            })
    return entities

def process_document(raw: str, icd: Ontology, rx: Ontology, llm_endpoint: str | None = None, llm_model: str | None = None, lab_aliases: set[str] | None = None, symptom_aliases: set[str] | None = None, require_llm: bool = True, llm_timeout: int = 180) -> list[dict]:
    if require_llm and not llm_endpoint:
        raise RuntimeError("LLM endpoint is required; pass --llm-endpoint or explicitly use --allow-rule-only")
    items = detect(raw, icd, rx, lab_aliases, symptom_aliases)
    if llm_endpoint:
        items.extend(align_llm(raw, llm_extract(raw, llm_endpoint, llm_model, llm_timeout)))
    resolved = resolve(raw, items)
    link_entities(resolved, icd, rx)
    return [e.public() for e in resolved if e.text == raw[e.start:e.end]]

def validate_pair(raw: str, data: list[dict]) -> list[str]:
    errors=[]
    for i,e in enumerate(data):
        if not isinstance(e,dict): errors.append(f"{i}: not object"); continue
        if e.get("type") not in ENTITY_TYPES: errors.append(f"{i}: invalid type")
        p=e.get("position");
        if not isinstance(p,list) or len(p)!=2 or not all(isinstance(x,int) for x in p) or not (0<=p[0]<p[1]<=len(raw)): errors.append(f"{i}: invalid position"); continue
        if raw[p[0]:p[1]] != e.get("text"): errors.append(f"{i}: text/position mismatch")
        if e.get("type") in ASSERTION_TYPES and any(x not in ALLOWED_ASSERTIONS for x in e.get("assertions",[])): errors.append(f"{i}: invalid assertion")
        if e.get("type") in {"CHẨN_ĐOÁN","THUỐC"} and not isinstance(e.get("candidates",[]),list): errors.append(f"{i}: candidates not list")
    return errors

def main(argv=None):
    ap=argparse.ArgumentParser(description="Clinical NLP contest inference")
    ap.add_argument("--input", default="input"); ap.add_argument("--output", default="output")
    ap.add_argument("--icd", required=True, help="versioned ICD-10 JSONL/CSV index")
    ap.add_argument("--rxnorm", required=True, help="versioned RxNorm JSONL/CSV index")
    ap.add_argument("--lab-dictionary", help="optional one-term-per-line lab/test aliases")
    ap.add_argument("--symptom-dictionary", help="optional one-term-per-line symptom aliases")
    ap.add_argument("--llm-endpoint", help="localhost OpenAI-compatible base URL (required unless --allow-rule-only)")
    ap.add_argument("--llm-model"); ap.add_argument("--llm-timeout", type=int, default=180, help="per-document localhost LLM timeout in seconds")
    ap.add_argument("--zip", action="store_true"); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--allow-rule-only", action="store_true", help="explicitly disable LLM (debug/calibration only; not contest mode)")
    args=ap.parse_args(argv); logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    icd=Ontology(args.icd, kind="icd"); rx=Ontology(args.rxnorm, kind="rxnorm")
    labs=load_terms(args.lab_dictionary); symptoms=load_terms(args.symptom_dictionary); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    files=sorted(Path(args.input).glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    for p in files:
        raw=p.read_text(encoding="utf-8"); data=process_document(raw,icd,rx,args.llm_endpoint,args.llm_model,labs,symptoms,not args.allow_rule_only,args.llm_timeout)
        (outdir/(p.stem+".json")).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        if args.validate:
            err=validate_pair(raw,data)
            if err: logging.error("%s: %s",p.name,"; ".join(err[:5]))
    if args.zip:
        import zipfile
        z=Path("output.zip")
        with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as f:
            for p in sorted(outdir.glob("*.json")): f.write(p, "output/"+p.name)
        logging.info("wrote %s",z)

if __name__ == "__main__": main()
