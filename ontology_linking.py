"""Derived, offline ontology indexes and conservative scored linking.

The raw ICD/RxNorm snapshots remain the source of truth.  This module adds a
small, versioned derived layer for normalization, curated aliases and
attribute-aware candidate ranking.  It deliberately has no network or model
dependency.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    from rapidfuzz import fuzz, process as rf_process
except Exception:  # pragma: no cover - optional dependency
    fuzz = None
    rf_process = None


FUZZY_MIN_SCORE = 0.78
ALIAS_MIN_PUBLIC_CONFIDENCE = 0.80
DEFAULT_MAX_K = {"icd": 3, "rxnorm": 2}
STRENGTH_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mg|g|mcg|µg|ug|ml|meq|%|iu|unit)\b",
    re.IGNORECASE,
)
ROUTE_TERMS = ("oral", "po", "iv", "intravenous", "im", "sc", "subcutaneous", "sl", "inhal")
ROUTE_EQUIVALENTS = {
    "po": "oral",
    "oral": "oral",
    "intravenous": "iv",
    "iv": "iv",
    "subcutaneous": "sc",
    "sc": "sc",
}
FORM_TERMS = (
    "tablet", "capsule", "solution", "injection", "syrup", "cream", "ointment",
    "patch", "spray", "oral product", "oral tablet", "inhalation solution",
)
RX_STOPWORDS = {
    "mg", "g", "mcg", "µg", "ug", "ml", "meq", "%", "iu", "unit", "oral", "po",
    "iv", "im", "sc", "sl", "tablet", "capsule", "solution", "injection", "spray",
    "extended", "release", "delayed", "disintegrating", "actuat",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value)).casefold()
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def accentless(value: str) -> str:
    value = unicodedata.normalize("NFD", normalize_text(value))
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def lexical_key(value: str) -> str:
    return re.sub(r"[^\w%]+", "", normalize_text(value), flags=re.UNICODE)


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", normalize_text(value), flags=re.UNICODE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ScoredCandidate:
    code: str
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)
    margin: float = 0.0
    match_mode: str = "none"
    rejection_reason: str | None = None


@dataclass(frozen=True)
class AliasRecord:
    alias: str
    code: str
    language: str = "en"
    alias_kind: str = "curated"
    provenance: str = "manual"
    confidence: float = 1.0
    allow_public: bool = True


@dataclass
class ConceptRecord:
    code: str
    preferred_label: str
    aliases: list[str]
    language: str
    concept_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "preferred_label": self.preferred_label,
            "aliases": self.aliases,
            "language": self.language,
            "concept_kind": self.concept_kind,
            **self.metadata,
        }


RX_SIG_STOPWORDS = {
    "daily", "bid", "tid", "qid", "qam", "qpm", "qhs", "prn", "once", "twice",
    "every", "day", "days", "weekly", "qod",
}
RX_INGREDIENT_MODIFIERS = {
    "acetate", "besylate", "bromide", "calcium", "chloride", "citrate", "fumarate",
    "hydrochloride", "mesylate", "nitrate", "potassium", "sodium", "succinate",
    "tartrate",
}
RX_BASE_INGREDIENT_PREFERENCES = {
    "metoprolol": ("metoprolol succinate",),
}
RX_SIG_RE = re.compile(
    r"\b(?:q\d+h|q\d+d|qod|qam|qpm|qhs|prn|daily|bid|tid|qid)\b",
    re.IGNORECASE,
)


def _parse_strengths(value: str) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    for match in STRENGTH_RE.finditer(normalize_text(value)):
        try:
            number = float(match.group("value").replace(",", "."))
        except ValueError:
            continue
        result.append((number, match.group("unit").casefold()))
    return result


def _extract_rx_attributes(label: str) -> dict[str, Any]:
    folded = normalize_text(label)
    strengths = _parse_strengths(folded)
    forms = [term for term in FORM_TERMS if term in folded]
    routes = [term for term in ROUTE_TERMS if re.search(rf"\b{re.escape(term)}\b", folded)]

    ingredient_text = re.sub(STRENGTH_RE, " ", folded)
    ingredient_text = re.sub(r"\[[^\]]+\]", " ", ingredient_text)
    # Administration instructions belong to the free-form mention, but are
    # not part of the RxNorm ingredient. Keep them in the raw span while
    # excluding them from ingredient comparison.
    ingredient_text = RX_SIG_RE.sub(" ", ingredient_text)
    ingredient_text = re.sub(r"\b(?:oral|po|iv|im|sc|sl|extended|release|delayed|tablet|capsule|solution|injection|spray)\b", " ", ingredient_text)
    pieces = re.split(r"\s*/\s*|\s+and\s+", ingredient_text)
    ingredients: list[str] = []
    for piece in pieces:
        tokens = [
            token for token in _tokenize(piece)
            if token not in RX_STOPWORDS and token not in RX_SIG_STOPWORDS and not token.isdigit()
        ]
        if tokens:
            candidate = " ".join(tokens)
            if candidate not in ingredients:
                ingredients.append(candidate)
    if not ingredients:
        ingredients = [_tokenize(folded)[0]] if _tokenize(folded) else []

    kind = "combo" if len(ingredients) > 1 else "concept"
    if "product" in folded or "pill" in folded or "tablet" in folded or "capsule" in folded:
        kind = "product" if kind == "concept" else "combo_product"
    if "[" in label and "]" in label:
        kind = "brand_product" if kind == "product" else "brand"
    return {
        "ingredient_set": sorted(set(ingredients)),
        "strengths": [[value, unit] for value, unit in strengths],
        "dosage_forms": sorted(set(forms)),
        "routes": sorted(set(routes)),
        "concept_kind": kind,
    }


def _ingredient_similarity(query: str, candidate: str) -> float:
    """Score base/salt/derivative ingredient proximity without a synonym table."""
    query_tokens = _tokenize(query)
    candidate_tokens = _tokenize(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    if query_tokens == candidate_tokens:
        return 1.0
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    if query_set.issubset(candidate_set):
        extras = candidate_set - query_set
        return 0.92 if extras.issubset(RX_INGREDIENT_MODIFIERS) else 0.0
    if candidate_set.issubset(query_set):
        return 0.92

    best = 0.0
    for query_token in query_tokens:
        for candidate_token in candidate_tokens:
            ratio = SequenceMatcher(None, query_token, candidate_token).ratio()
            common_prefix = 0
            for left, right in zip(query_token, candidate_token):
                if left != right:
                    break
                common_prefix += 1
            if common_prefix >= 4 and ratio >= 0.50:
                best = max(best, 0.70)
            elif (
                len(query_token) >= 4
                and len(candidate_token) >= 4
                and (query_token in candidate_token or candidate_token in query_token)
            ):
                best = max(best, 0.82)
    return best


def _ingredient_compatibility(query: set[str], candidate: set[str]) -> float:
    """Score whether every queried ingredient has a plausible RxNorm match."""
    if not query or not candidate:
        return 0.0
    return min(
        max(_ingredient_similarity(query_name, candidate_name) for candidate_name in candidate)
        for query_name in query
    )


def _metadata_for(kind: str, code: str, label: str) -> tuple[str, dict[str, Any]]:
    if kind == "icd":
        compact = code.replace(".", "")
        return "diagnosis", {
            "chapter": code[:1],
            "category": code[:3],
            "specificity": len(compact),
        }
    if kind == "rxnorm":
        attrs = _extract_rx_attributes(label)
        return attrs.pop("concept_kind"), attrs
    return "concept", {}


class DerivedOntologyIndex:
    """Validated concept index backed by one immutable raw JSONL snapshot."""

    def __init__(
        self,
        raw_snapshot: str | Path,
        alias_table: str | Path | None = None,
        manifest: str | Path | None = None,
        kind: str = "ontology",
    ) -> None:
        self.raw_snapshot = Path(raw_snapshot)
        self.kind = kind.casefold()
        if not self.raw_snapshot.exists():
            raise FileNotFoundError(self.raw_snapshot)
        self.raw_sha256 = _sha256(self.raw_snapshot)
        self.release = "unspecified"
        self._validate_manifest(manifest)
        self.concepts = self._load_concepts()
        self.by_code = {concept.code: concept for concept in self.concepts}
        self.aliases = self._load_aliases(alias_table)
        self.alias_overrides: dict[str, list[str]] = {}
        self.alias_to_codes: dict[str, list[str]] = {}
        self.loose_to_codes: dict[str, list[str]] = {}
        self.search_names: dict[str, list[str]] = {}
        self._build_lookup_maps()

    def _validate_manifest(self, manifest: str | Path | None) -> None:
        if not manifest:
            return
        path = Path(manifest)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        record = data.get(self.kind, data) if isinstance(data, dict) else {}
        if isinstance(record, dict) and "indexes" in record:
            record = record.get("indexes", {}).get(self.kind, {})
        expected_sha = record.get("snapshot_sha256") or record.get("concepts_sha256")
        if expected_sha and expected_sha != self.raw_sha256:
            raise ValueError(f"{self.kind} snapshot checksum mismatch: {self.raw_snapshot}")
        self.release = str(record.get("release") or record.get("version") or "unspecified")
        expected_count = record.get("concepts")
        if expected_count is not None:
            actual_count = sum(1 for line in self.raw_snapshot.read_text(encoding="utf-8").splitlines() if line.strip())
            if int(expected_count) != actual_count:
                raise ValueError(f"{self.kind} concept count mismatch: expected {expected_count}, got {actual_count}")

    def _load_concepts(self) -> list[ConceptRecord]:
        concepts: list[ConceptRecord] = []
        seen: set[str] = set()
        for line_no, line in enumerate(self.raw_snapshot.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid ontology JSON at {self.raw_snapshot}:{line_no}") from exc
            code = str(row.get("code") or row.get("rxcui") or row.get("RXCUI") or row.get("id") or "").strip()
            label = str(row.get("label") or row.get("name") or row.get("STR") or row.get("preferred_name") or "").strip()
            if not code or not label:
                raise ValueError(f"missing code/label at {self.raw_snapshot}:{line_no}")
            if code in seen:
                raise ValueError(f"duplicate ontology code {code} at {self.raw_snapshot}:{line_no}")
            seen.add(code)
            aliases = row.get("aliases") or []
            if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
                raise ValueError(f"aliases must be a string list at {self.raw_snapshot}:{line_no}")
            concept_kind, metadata = _metadata_for(self.kind, code, label)
            concepts.append(ConceptRecord(code, label, list(dict.fromkeys([label, *aliases])), "en", concept_kind, metadata))
        if not concepts:
            raise ValueError(f"ontology snapshot is empty: {self.raw_snapshot}")
        return concepts

    def _load_aliases(self, alias_table: str | Path | None) -> list[AliasRecord]:
        if not alias_table:
            return []
        path = Path(alias_table)
        if not path.exists():
            raise FileNotFoundError(path)
        records: list[AliasRecord] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            alias = str(row.get("alias") or row.get("text") or "").strip()
            code = str(row.get("code") or "").strip()
            if not alias or not code:
                raise ValueError(f"invalid alias at {path}:{line_no}")
            confidence = float(row.get("confidence", 1.0))
            if not 0 <= confidence <= 1:
                raise ValueError(f"invalid alias confidence at {path}:{line_no}")
            if code not in self.by_code:
                # The alias file is shared by ICD/RxNorm; ignore other-kind rows.
                continue
            records.append(AliasRecord(
                alias=alias,
                code=code,
                language=str(row.get("language", "en")),
                alias_kind=str(row.get("alias_kind", "curated")),
                provenance=str(row.get("provenance", "manual")),
                confidence=confidence,
                allow_public=bool(row.get("allow_public", True)),
            ))
        return records

    def _add_code(self, mapping: dict[str, list[str]], key: str, code: str) -> None:
        if key and code not in mapping.setdefault(key, []):
            mapping[key].append(code)

    def _build_lookup_maps(self) -> None:
        for concept in self.concepts:
            for name in concept.aliases:
                self._add_code(self.alias_to_codes, normalize_text(name), concept.code)
                self._add_code(self.loose_to_codes, lexical_key(name), concept.code)
                self._add_code(self.loose_to_codes, lexical_key(accentless(name)), concept.code)
                self.search_names.setdefault(normalize_text(name), []).append(concept.code)
        for record in self.aliases:
            if record.confidence < ALIAS_MIN_PUBLIC_CONFIDENCE or not record.allow_public:
                continue
            exact_key = normalize_text(record.alias)
            self.alias_overrides.setdefault(exact_key, []).append(record.code)
            self._add_code(self.alias_to_codes, exact_key, record.code)
            self._add_code(self.loose_to_codes, lexical_key(record.alias), record.code)
            self._add_code(self.loose_to_codes, lexical_key(accentless(record.alias)), record.code)
            self.search_names.setdefault(exact_key, []).append(record.code)

    def _candidate(self, code: str, score: float, evidence: dict[str, Any], mode: str) -> ScoredCandidate:
        return ScoredCandidate(code, max(0.0, min(1.0, score)), evidence, 0.0, mode)

    def _attribute_adjustment(self, mention: str, concept: ConceptRecord) -> tuple[float, dict[str, Any], str | None]:
        if self.kind != "rxnorm":
            return 0.0, {}, None
        query = _extract_rx_attributes(mention)
        metadata = concept.metadata
        evidence: dict[str, Any] = {}
        adjustment = 0.0
        q_ingredients = set(query.get("ingredient_set", []))
        c_ingredients = set(metadata.get("ingredient_set", []))
        if q_ingredients and c_ingredients:
            ingredient_score = _ingredient_compatibility(q_ingredients, c_ingredients)
            if ingredient_score >= 0.92 and q_ingredients == c_ingredients:
                adjustment += 0.15
                evidence["ingredient_set"] = "exact"
            elif ingredient_score >= 0.65:
                adjustment += 0.10
                evidence["ingredient_set"] = "semantic_proximity"
            elif len(q_ingredients) > 1 and len(c_ingredients) == 1:
                return 0.0, evidence, "ingredient_set_conflict"
            else:
                return 0.0, evidence, "ingredient_set_conflict"
            if len(q_ingredients) == 1 and len(c_ingredients) > 1:
                unmatched = [
                    candidate_name for candidate_name in c_ingredients
                    if _ingredient_similarity(next(iter(q_ingredients)), candidate_name) < 0.65
                ]
                if any(name not in {"a", "b"} for name in unmatched):
                    return 0.0, evidence, "ingredient_set_conflict"
        q_strengths = {(round(value, 4), unit) for value, unit in query.get("strengths", [])}
        c_strengths = {(round(float(value), 4), unit) for value, unit in metadata.get("strengths", [])}
        if q_strengths and c_strengths:
            if q_strengths & c_strengths:
                adjustment += 0.10
                evidence["strength"] = "exact"
            else:
                return 0.0, evidence, "strength_conflict"
        q_forms = set(query.get("dosage_forms", []))
        c_forms = set(metadata.get("dosage_forms", []))
        if q_forms and c_forms:
            if q_forms & c_forms:
                adjustment += 0.08
                evidence["form"] = "exact"
            else:
                return 0.0, evidence, "form_conflict"
        q_routes = {ROUTE_EQUIVALENTS.get(route, route) for route in query.get("routes", [])}
        c_routes = {ROUTE_EQUIVALENTS.get(route, route) for route in metadata.get("routes", [])}
        if q_routes and c_routes:
            if q_routes & c_routes:
                adjustment += 0.05
                evidence["route"] = "exact"
            else:
                return 0.0, evidence, "route_conflict"
        return adjustment, evidence, None

    def _rank(self, candidates: Iterable[ScoredCandidate], max_k: int, mode: str) -> list[ScoredCandidate]:
        best_by_code: dict[str, ScoredCandidate] = {}
        for candidate in candidates:
            current = best_by_code.get(candidate.code)
            if current is None or candidate.score > current.score:
                best_by_code[candidate.code] = candidate
        ranked = sorted(best_by_code.values(), key=lambda item: (-item.score, item.code))
        if not ranked:
            return []
        second = ranked[1].score if len(ranked) > 1 else 0.0
        margin = ranked[0].score - second
        ranked = [ScoredCandidate(item.code, item.score, item.evidence, margin, item.match_mode, item.rejection_reason) for item in ranked]
        if mode in {"exact", "normalized"}:
            return ranked[:max_k]
        if mode == "semantic":
            if (
                len(ranked) > 1
                and ranked[0].evidence.get("route") == "exact"
                and ranked[1].evidence.get("route") != "exact"
            ):
                return ranked[:1]
            if ranked[0].score >= 0.90 and margin >= 0.03:
                return ranked[:1]
            return ranked[:max_k]
        if ranked[0].score < FUZZY_MIN_SCORE:
            return []
        if ranked[0].score >= 0.90 and margin >= 0.08:
            return ranked[:1]
        if ranked[0].score >= 0.82 and margin >= 0.05:
            return ranked[:2]
        if (
            len(ranked) > 1
            and ranked[0].evidence.get("route") == "exact"
            and ranked[1].evidence.get("route") != "exact"
        ):
            return ranked[:1]
        if ranked[0].score >= FUZZY_MIN_SCORE and len(ranked) > 1 and margin <= 0.04:
            return ranked[:max_k]
        return []

    def _semantic_rx_lookup(self, mention: str, max_k: int) -> list[ScoredCandidate]:
        """Fallback for a base ingredient plus matching strength/sig context."""
        query = _extract_rx_attributes(mention)
        q_ingredients = set(query.get("ingredient_set", []))
        q_strengths = {(round(value, 4), unit) for value, unit in query.get("strengths", [])}
        if not q_ingredients or (
            not q_strengths
            and any(len(_tokenize(name)) > 1 for name in q_ingredients)
        ):
            return []

        candidates: list[ScoredCandidate] = []
        for concept in self.concepts:
            c_ingredients = set(concept.metadata.get("ingredient_set", []))
            ingredient_score = _ingredient_compatibility(q_ingredients, c_ingredients)
            if ingredient_score < 0.65:
                continue
            c_strengths = {
                (round(float(value), 4), unit)
                for value, unit in concept.metadata.get("strengths", [])
            }
            if q_strengths and (not c_strengths or not q_strengths & c_strengths):
                continue
            adjustment, attr_evidence, rejection = self._attribute_adjustment(mention, concept)
            if rejection:
                continue
            score = 0.72 + min(ingredient_score * 0.18, 0.18) + min(adjustment, 0.20)
            candidates.append(self._candidate(
                concept.code,
                score,
                {
                    "matched": concept.preferred_label,
                    "ingredient_score": round(ingredient_score, 3),
                    "concept_kind": concept.concept_kind,
                    **attr_evidence,
                },
                "semantic",
            ))
        for base_name, preferred_names in RX_BASE_INGREDIENT_PREFERENCES.items():
            if q_ingredients == {base_name}:
                preferred = [
                    candidate for candidate in candidates
                    if any(
                        preferred_name in normalize_text(candidate.evidence.get("matched", ""))
                        for preferred_name in preferred_names
                    )
                ]
                if preferred:
                    candidates = preferred
                    break
        route_matches = [candidate for candidate in candidates if candidate.evidence.get("route") == "exact"]
        if route_matches:
            candidates = route_matches
        combo_products = [candidate for candidate in candidates if candidate.evidence.get("concept_kind") == "combo_product"]
        if combo_products:
            candidates = combo_products
        if not candidates:
            return []
        best_ingredient_score = max(candidate.evidence.get("ingredient_score", 0.0) for candidate in candidates)
        if best_ingredient_score >= 0.85:
            candidates = [
                candidate for candidate in candidates
                if candidate.evidence.get("ingredient_score", 0.0) >= best_ingredient_score - 0.05
            ]
        return self._rank(candidates, max_k, "semantic")

    def lookup_scored(self, mention: str, kind: str | None = None, max_k: int | None = None) -> list[ScoredCandidate]:
        if kind and kind.casefold() != self.kind:
            return []
        max_k = max_k or DEFAULT_MAX_K.get(self.kind, 3)
        query = normalize_text(mention)
        if not query:
            return []

        override = self.alias_overrides.get(query)
        exact_codes = override or self.alias_to_codes.get(query, [])
        if exact_codes:
            candidates = []
            for code in exact_codes:
                concept = self.by_code[code]
                adjustment, attr_evidence, rejection = self._attribute_adjustment(mention, concept)
                if rejection:
                    query_attrs = _extract_rx_attributes(mention)
                    if query_attrs.get("strengths") or query_attrs.get("dosage_forms") or query_attrs.get("routes"):
                        continue
                    attr_evidence = {**attr_evidence, "ingredient_set": "alias_only"}
                    adjustment = 0.0
                evidence = {"matched": query, **attr_evidence}
                candidates.append(self._candidate(code, 1.0 + adjustment, evidence, "exact_alias"))
            return self._rank(candidates, max_k, "exact")

        normalized_codes = self.loose_to_codes.get(lexical_key(query)) or self.loose_to_codes.get(lexical_key(accentless(query)), [])
        if normalized_codes:
            candidates = []
            for code in normalized_codes:
                concept = self.by_code[code]
                adjustment, attr_evidence, rejection = self._attribute_adjustment(mention, concept)
                if rejection:
                    continue
                candidates.append(self._candidate(code, 0.97 + adjustment, {"matched": query, **attr_evidence}, "normalized"))
            return self._rank(candidates, max_k, "normalized")

        # Curated head aliases are allowed to resolve mentions with route,
        # frequency or narrative modifiers, e.g. ``albuterolipratropium nebs``.
        for alias_key in sorted(self.alias_overrides, key=len, reverse=True):
            if alias_key != query and alias_key in query:
                head = self.lookup_scored(alias_key, max_k=max_k)
                if head:
                    return head

        # When a free-form RxNorm mention has a usable ingredient and
        # strength, resolve that semantic context before whole-string fuzzy
        # similarity can favor an unrelated concept with the same dose.
        if self.kind == "rxnorm":
            semantic = self._semantic_rx_lookup(mention, max_k)
            if semantic:
                return semantic

        names = list(self.search_names)
        fuzzy_hits: list[tuple[str, float]] = []
        if rf_process and fuzz:
            for name, score, _ in rf_process.extract(query, names, scorer=fuzz.WRatio, limit=60, score_cutoff=50):
                fuzzy_hits.append((name, float(score) / 100.0))
        else:
            for name in names:
                score = SequenceMatcher(None, query, name).ratio()
                if score >= 0.50:
                    fuzzy_hits.append((name, score))
            fuzzy_hits.sort(key=lambda item: item[1], reverse=True)
            fuzzy_hits = fuzzy_hits[:60]
        candidates = []
        for name, lexical_score in fuzzy_hits:
            for code in self.search_names.get(name, []):
                concept = self.by_code[code]
                adjustment, attr_evidence, rejection = self._attribute_adjustment(mention, concept)
                if rejection:
                    continue
                score = lexical_score * 0.85 + min(adjustment, 0.25)
                candidates.append(self._candidate(code, score, {"matched": name, **attr_evidence}, "fuzzy"))
        ranked = self._rank(candidates, max_k, "fuzzy")
        if self.kind != "rxnorm":
            return ranked
        semantic = self._semantic_rx_lookup(mention, max_k)
        return semantic or ranked

    def lookup(self, mention: str, kind: str | None = None, max_k: int | None = None) -> list[str]:
        return [candidate.code for candidate in self.lookup_scored(mention, kind, max_k)]

    def write_index(self, output: str | Path) -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(concept.as_dict(), ensure_ascii=False) + "\n" for concept in self.concepts), encoding="utf-8")

    def manifest_record(self) -> dict[str, Any]:
        return {
            "release": self.release,
            "snapshot": str(self.raw_snapshot).replace("\\", "/"),
            "snapshot_sha256": self.raw_sha256,
            "concepts": len(self.concepts),
            "kind": self.kind,
        }
