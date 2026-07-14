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

class Ontology:
    def __init__(self, path: str | None, kind: str = "ontology"):
        self.kind, self.rows = kind, []
        if path:
            if not Path(path).exists(): raise FileNotFoundError(path)
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
        except Exception as e: logging.warning("ontology load failed %s: %s", p, e)

    def lookup(self, mention: str, k: int = 3) -> list[str]:
        q = norm(mention); exact = self.alias_to_codes.get(q)
        if exact: return exact[:k]
        q2 = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ml|%|meq)\b", "", q)
        q2 = re.sub(r"\b(?:po|oral|iv|im|sc|sl|daily|bid|tid|qid|qam|qhs|prn|q\d+h)\b", "", q2).strip()
        scored = []
        tokens = set(re.findall(r"[\wÀ-ỹ]+", q2))
        pool = []
        for token in tokens: pool.extend(self.rows_by_first.get(token, []))
        for n, codes in pool or self.rows:
            a = fuzz.ratio(q2, norm(n)) / 100 if fuzz else (1.0 if q2 in norm(n) or norm(n) in q2 else 0.0)
            if a >= .52: scored.append((a, codes[0]))
        return [c for _, c in sorted(scored, reverse=True)[:k]]

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
    a = max(raw.rfind(".", 0, start), raw.rfind("\n", 0, start), raw.rfind(";", 0, start)) + 1
    b0 = [x for x in (raw.find(".", end), raw.find("\n", end), raw.find(";", end)) if x >= 0]
    b = min(b0) if b0 else len(raw)
    return a, b, raw[a:b]

def section_historical(raw: str, pos: int) -> bool:
    before = raw[max(0, raw.rfind("\n", 0, pos) - 500):pos].casefold()
    current = any(x in before for x in ("bệnh sử hiện tại", "triệu chứng hiện tại", "lý do nhập viện", "tình trạng lúc vào viện"))
    if current: return False
    return any(x in before for x in ("tiền sử", "thuốc trước khi nhập viện", "bệnh đã điều trị trước đây", "lịch sử bệnh"))

def assertions(raw: str, e: Entity) -> list[str]:
    if e.typ not in ASSERTION_TYPES: return []
    a, b, sent = sentence_window(raw, e.start, e.end); s = norm(sent)
    out = []
    neg = ("không", "chưa", "âm tính", "không có", "không ghi nhận", "không phát hiện", "không thấy", "phủ nhận", "không bằng chứng")
    fam = ("bố bệnh nhân", "mẹ bệnh nhân", "cha bệnh nhân", "anh chị em", "người nhà", "gia đình", "họ hàng", "mẹ ", "bố ")
    hist = ("tiền sử", "đã từng", "trước đây", "đã điều trị", "ngừng", "ngừng uống", "trước khi nhập viện", "đã dùng")
    if any(x in s for x in neg): out.append("isNegated")
    if any(x in s for x in fam): out.append("isFamily")
    if section_historical(raw, e.start) or any(x in s for x in hist): out.append("isHistorical")
    return [x for x in ALLOWED_ASSERTIONS if x in out]

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
            trimmed = re.sub(r"[\s,;:.-]+$", "", raw[a:b])
            b = a + len(trimmed)
            add_match(out, raw, a, b, "THUỐC", .9, rx.lookup(raw[a:b], 2))
    # Generic medication line fallback: a likely name plus route/dose or known med cue.
    for m in re.finditer(r"(?im)(?<!\w)([A-Za-z][A-Za-z0-9+/'-]{2,}(?:\s+[A-Za-z][A-Za-z0-9+/'-]{1,}){0,3})(?=\s+(?:\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ml|meq)|po|oral|iv|daily|bid|qid|prn)\b)", raw):
        if norm(m.group(1)) not in {norm(x.text) for x in out if x.typ == "THUỐC"}:
            add_match(out, raw, m.start(), m.end(), "THUỐC", .62, rx.lookup(m.group(1), 2))
    # Tests and result values: only numbers/words in lab context.
    lab_names = sorted(lab_aliases, key=len, reverse=True)
    for name in lab_names:
        for a, b in find_all(raw, name):
            add_match(out, raw, a, b, "TÊN_XÉT_NGHIỆM", .8)
            tail = raw[b:min(len(raw), b + 90)]
            vm = re.match(r"\s*(?::|=|là|tăng(?:\s+nhẹ)?\s+lên|giảm(?:\s+nhẹ)?\s+xuống)?\s*([+-]?\d+(?:[.,]\d+)?(?:\s*(?:mg/dl|g/dl|%|mmol/l|umol/l|ng/ml))?|âm tính|dương tính|thấp|cao)", tail, re.I)
            if vm: add_match(out, raw, b + vm.start(1), b + vm.end(1), "KẾT_QUẢ_XÉT_NGHIỆM", .82)
    # Generic lab marker:value (avoid drug dose because nearby medication wins).
    for m in re.finditer(r"(?i)\b([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9%_ /-]{1,35})\s*[:=]\s*([+-]?\d+(?:[.,]\d+)?(?:\s*[A-Za-z%/µ]+)?|âm tính|dương tính)", raw):
        key = norm(m.group(1));
        if key in {norm(x) for x in lab_aliases}:
            add_match(out, raw, m.start(1), m.end(1), "TÊN_XÉT_NGHIỆM", .75)
            add_match(out, raw, m.start(2), m.end(2), "KẾT_QUẢ_XÉT_NGHIỆM", .75)
    # Symptoms and diagnoses expressed only in clinical prose.
    for phrase in sorted(symptom_aliases, key=len, reverse=True):
        for a, b in find_all(raw, phrase): add_match(out, raw, a, b, "TRIỆU_CHỨNG", .7)
    return out

def resolve(raw: str, items: list[Entity]) -> list[Entity]:
    # Merge exact span/type, then remove nested weaker duplicates. Keep repeated occurrences.
    merged: dict[tuple[int,int,str], Entity] = {}
    for e in items:
        k = (e.start, e.end, e.typ)
        if k not in merged or e.confidence > merged[k].confidence:
            merged[k] = e
        else:
            merged[k].candidates = list(dict.fromkeys(merged[k].candidates + e.candidates))
    vals = list(merged.values())
    keep = []
    for e in sorted(vals, key=lambda x: (x.start, -(x.end-x.start), -x.confidence)):
        blocked = False
        for q in keep:
            if q.start <= e.start and e.end <= q.end and (q.end-q.start) > (e.end-e.start) and q.typ == e.typ:
                blocked = True; break
        if not blocked: keep.append(e)
    for e in keep:
        e.assertions = list(dict.fromkeys(assertions(raw, e) + e.assertions))
        if e.typ in {"CHẨN_ĐOÁN", "THUỐC"}: e.candidates = list(dict.fromkeys(e.candidates))
    return sorted(keep, key=lambda x: (x.start, x.end, x.typ))

def llm_extract(raw: str, endpoint: str | None, model: str | None, timeout: int = 45) -> list[dict]:
    if not endpoint: return []
    logging.info("LLM extraction request: endpoint=%s model=%s chars=%d", endpoint, model or "local", len(raw))
    prompt = ("Extract Vietnamese clinical entities. Return JSON array only, each item "
              "{text,type,assertions}. Copy text exactly. Types: TRIỆU_CHỨNG,TÊN_XÉT_NGHIỆM,"
              "KẾT_QUẢ_XÉT_NGHIỆM,CHẨN_ĐOÁN,THUỐC. Do not return positions. INPUT:\n" + raw)
    payload = {"model": model or "local", "messages":[{"role":"system","content":"You are a clinical NLP extraction module. Return JSON only."},{"role":"user","content":prompt}], "temperature":0, "max_tokens":1800}
    qwen_payload = dict(payload)
    qwen_payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        base = endpoint.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        data = None
        last_error = None
        for body_obj in (qwen_payload, payload):
            body = json.dumps(body_obj).encode()
            req = urllib.request.Request(url, body, {"Content-Type":"application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r: data = json.loads(r.read().decode())
                break
            except Exception as e:
                last_error = e
                if body_obj is payload:
                    raise
                logging.warning("LLM rejected Qwen-specific payload; retrying OpenAI-minimal payload: %s", e)
        if data is None: raise RuntimeError(last_error or "empty LLM response")
        text = data["choices"][0]["message"]["content"]
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < start: raise RuntimeError("LLM response does not contain a JSON array")
        value = json.loads(text[start:end+1])
        if not isinstance(value, list): raise RuntimeError("LLM response JSON must be an array")
        logging.info("LLM extraction response: %d items", len(value))
        return value
    except Exception as e:
        raise RuntimeError(f"localhost LLM request/parse failed: {e}") from e

def align_llm(raw: str, items: list[dict]) -> list[Entity]:
    out=[]
    for x in items:
        text, typ = str(x.get("text", "")), x.get("type")
        if typ not in ENTITY_TYPES or not text: continue
        for a,b in find_all(raw, text):
            e=Entity(raw[a:b], typ, a,b, assertions=[z for z in x.get("assertions",[]) if z in ALLOWED_ASSERTIONS], confidence=.55, source="llm"); out.append(e); break
    return out

def process_document(raw: str, icd: Ontology, rx: Ontology, llm_endpoint: str | None = None, llm_model: str | None = None, lab_aliases: set[str] | None = None, symptom_aliases: set[str] | None = None, require_llm: bool = True) -> list[dict]:
    if require_llm and not llm_endpoint:
        raise RuntimeError("LLM endpoint is required; pass --llm-endpoint or explicitly use --allow-rule-only")
    items = detect(raw, icd, rx, lab_aliases, symptom_aliases)
    if llm_endpoint:
        items.extend(align_llm(raw, llm_extract(raw, llm_endpoint, llm_model)))
    return [e.public() for e in resolve(raw, items) if e.text == raw[e.start:e.end]]

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
    ap.add_argument("--llm-model"); ap.add_argument("--zip", action="store_true"); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--allow-rule-only", action="store_true", help="explicitly disable LLM (debug/calibration only; not contest mode)")
    args=ap.parse_args(argv); logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    icd=Ontology(args.icd, kind="icd"); rx=Ontology(args.rxnorm, kind="rxnorm")
    labs=load_terms(args.lab_dictionary); symptoms=load_terms(args.symptom_dictionary); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    files=sorted(Path(args.input).glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    for p in files:
        raw=p.read_text(encoding="utf-8"); data=process_document(raw,icd,rx,args.llm_endpoint,args.llm_model,labs,symptoms,not args.allow_rule_only)
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
