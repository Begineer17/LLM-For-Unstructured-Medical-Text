## Phần nâng cấp theo research proposal: tích hợp end-to-end

Phần này chuyển các ý tưởng nghiên cứu trong proposal thành các module có thể gắn vào repository. Không phải paper nào cũng nên chạy trực tiếp trong inference.

```text
paper phù hợp ngôn ngữ + ontology + metric
        → module inference hoặc train chính

paper khác domain/ngôn ngữ
        → teacher, pseudo-label, hard-negative generator hoặc ablation

paper tập trung relation/temporal nhưng output không chấm relation
        → graph consistency nội bộ, không thêm field JSON
```

### 1. Bản đồ thay đổi từ pipeline hiện tại

| Module hiện tại | Nâng cấp đề xuất | File cần thêm/sửa | Vai trò runtime |
|---|---|---|---|
| `llm_extractor.py` | GUIDEX-style structured generation; OpenBioNER type descriptions | `synthetic/guidex.py`, `candidate_ner.py` | Chủ yếu offline/ambiguous fallback |
| `rules.py` + `sections.py` | MedCAT-style modular components, ConText scope | `context_components.py`, `assertions.py` | Chạy mọi document |
| `ontology.py` | BioLORD/SapBERT/Qwen embedding adapters | `retrievers/*.py` | Candidate retrieval |
| `linkers.py` | Prompt-BioEL-style listwise reranker | `reranker.py` | Chỉ chạy top-k nhỏ |
| candidate selection | ANGEL-style hard negatives + dynamic top-k | `hard_negatives.py`, `calibration.py` | Chọn candidate cuối |
| span resolver | character-level ensemble | `char_ensemble.py`, `boundary.py` | Tăng exact-span score |
| assertion resolver | 3 sigmoid heads + rule constraints | `assertion_model.py` | Multi-label assertion |
| concept graph nội bộ | GraphTREx-lite | `concept_graph.py`, `graph_constraints.py` | Consistency, không serialize relation |

### 2. OpenBioNER-v2: dùng đúng vai trò

OpenBioNER-v2 là mô hình NER điều kiện theo mô tả type. Các model công khai nằm ở [Hugging Face collection](https://huggingface.co/collections/disi-unibo-nlp/openbioner-v2), không nên ghi nhầm là một GitHub repository chính thức. Model card ghi language là English; tài liệu cũng nêu IOB không biểu diễn nested span và zero-shot model có thể sai boundary. Vì vậy, đây là **teacher/ablation model**, không phải Vietnamese production NER mặc định.

Nguồn tham khảo:

- [OpenBioNER-v2 model card](https://huggingface.co/disi-unibo-nlp/openbioner-base-v2)
- [OpenBioNER-v2 limitations and usage](https://huggingface.co/blog/alecocc/openbioner-v2)
- [OpenBioNER-v2 paper](https://aclanthology.org/2025.findings-naacl.47/)

#### 2.1. Adapter sinh candidate span

Thêm `src/clinical_nlp/candidate_ner.py`:

```python
from dataclasses import dataclass

@dataclass
class SpanCandidate:
    start: int
    end: int
    text: str
    entity_type: str
    score: float
    source: str

TYPE_DESCRIPTIONS = {
    "TRIỆU_CHỨNG": "A patient symptom or clinical manifestation, not a confirmed disease name.",
    "TÊN_XÉT_NGHIỆM": "The name of a laboratory test, measurement, marker or clinical examination.",
    "KẾT_QUẢ_XÉT_NGHIỆM": "A textual or numeric result belonging to a test, including unit when present.",
    "CHẨN_ĐOÁN": "A disease or condition diagnosed, suspected or stated as a diagnosis.",
    "THUỐC": "A drug, ingredient or clinical drug expression with optional strength and route.",
}

class DescriptionConditionedGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text: str) -> list[SpanCandidate]:
        """Model-specific token-span inference.

        Do not trust model offsets. Convert token spans to raw character spans
        and run the boundary resolver before creating public Entity objects.
        """
        raise NotImplementedError
```

Không truyền trực tiếp output của OpenBioNER vào JSON. Ghép candidate từ:

```text
OpenBioNER/type-description candidates
+ Vietnamese encoder candidates
+ drug/lab dictionaries
+ regex
        → raw-span alignment
        → boundary resolver
        → type resolver
        → public Entity
```

Nếu zero-shot English không phù hợp tiếng Việt, giữ nguyên ý tưởng description-conditioning nhưng fine-tune backbone XLM-R/PhoBERT/ViPubmedDeBERTa trên synthetic data của cuộc thi.

### 3. MedCAT v2: lấy kiến trúc modular bằng adapter

MedCAT v2 hữu ích ở việc tách component registry, NER, linker, MetaCAT và relation addon. Repository MedCAT cũ đã archived; dự án chuyển sang [CogStack/cogstack-nlp](https://github.com/CogStack/cogstack-nlp). Tài liệu kiến trúc v2 nằm tại [CogStack documentation](https://docs.cogstack.org/en/latest/medcat/medcat-v2/docs/architecture/).

Không dùng model pack MedCAT có sẵn như thể nó khớp cuộc thi: các pack phổ biến gắn với UMLS/SNOMED/MIMIC và khác language/ontology. Viết interface tương tự:

```python
class ClinicalComponent:
    name: str

    def process(self, doc, state):
        raise NotImplementedError

class MentionComponent(ClinicalComponent):
    name = "mention_candidates"

class LinkerComponent(ClinicalComponent):
    name = "icd_rxnorm_linker"

class AssertionComponent(ClinicalComponent):
    name = "vietnamese_context"

class GraphConstraintComponent(ClinicalComponent):
    name = "concept_graph_constraints"

COMPONENTS = [
    DocumentViewComponent(),
    SectionComponent(),
    MentionComponent(),
    SpanResolverComponent(),
    AssertionComponent(),
    LinkerComponent(),
    GraphConstraintComponent(),
    OutputValidatorComponent(),
]

def process(doc):
    state = {}
    for component in COMPONENTS:
        state = component.process(doc, state)
    return serialize_public_output(state)
```

`LinkerComponent` phải trả ICD code/RxCUI theo KB của BTC; `AssertionComponent` phải trả đúng `isNegated`, `isFamily`, `isHistorical`.

### 4. GUIDEX-style synthetic data: module nên đầu tư đầu tiên

GUIDEX tách synthetic IE thành các bước có cấu trúc thay vì yêu cầu LLM sinh một lần cả văn bản, nhãn và offset. Dùng [paper ACL 2025](https://aclanthology.org/2025.findings-acl.1245/) và [official repository HiTZ/GUIDEX](https://github.com/HiTZ/GUIDEX) làm thiết kế tham khảo. Repository README hiện có pipeline generation và notebook NER; cần thay model/prompt bằng dữ liệu y khoa tiếng Việt của đội.

#### 4.1. Structured record

Thêm `src/clinical_nlp/synthetic/guidex.py`:

```python
from dataclasses import dataclass

@dataclass
class SyntheticEntity:
    entity_id: str
    surface: str
    entity_type: str
    assertions: list[str]
    candidates: list[str]

@dataclass
class StructuredClinicalRecord:
    document_style: str
    sections: list[str]
    entities: list[SyntheticEntity]
    relations: list[dict]

def build_record(concepts, style, seed):
    """Sample a record from ICD/RxNorm/lab dictionaries.

    The record is the source of truth; the LLM cannot invent codes or offsets.
    """
    raise NotImplementedError
```

Ví dụ record:

```json
{
  "document_style": "discharge_note_vi",
  "sections": ["Tiền sử bệnh", "Bệnh sử hiện tại", "Đánh giá tại bệnh viện"],
  "entities": [
    {"entity_id":"E1","surface":"đau thượng vị","entity_type":"TRIỆU_CHỨNG",
     "assertions":[],"candidates":[]},
    {"entity_id":"E2","surface":"trào ngược dạ dày - thực quản","entity_type":"CHẨN_ĐOÁN",
     "assertions":[],"candidates":["K21.0","K21.9"]},
    {"entity_id":"E3","surface":"omeprazole 20 mg","entity_type":"THUỐC",
     "assertions":["isHistorical"],"candidates":[]}
  ],
  "relations": [{"source":"E2","type":"CO_OCCURS_WITH","target":"E1"}]
}
```

#### 4.2. Marker rendering và offset tính bằng code

```text
structured record
        ↓
LLM render văn bản có marker
        ↓
parser strip marker
        ↓
exact raw spans
        ↓
rules + second-model validation
        ↓
accepted.jsonl
```

Ví dụ marker:

```text
<ENT id="E1" type="TRIỆU_CHỨNG">đau thượng vị</ENT>
```

Parser phải từ chối marker lồng sai, entity ID không tồn tại, type ngoài schema và marker không khớp surface của record. Sau khi strip:

```python
assert raw_text[start:end] == surface
```

Lưu provenance:

```json
{"sample_id":"syn_000123","teacher":"qwen3-8b",
 "record_hash":"...","checks":["span_exact","type_allowed"],
 "accepted":true}
```

#### 4.3. Các biến thể cần sinh

```text
discharge note / doctor note / medication list / lab report
phủ định / family / historical
14,43 và 14.43; mg, mg/ml, mmol/L, G/L
viết tắt, thiếu dấu, dính từ, code-switching
thuốc có hoặc không có strength
ICD specified/unspecified và các sibling code
```

Nên sinh 20.000–50.000 record, nhưng chỉ train record đã qua validator và provenance filter. Không dùng synthetic validation làm dev score duy nhất.

### 5. TransFusion: chỉ dùng trong data factory

TransFusion nghiên cứu dịch dữ liệu low-resource sang English rồi fusion annotation; paper ở [ACL 2025](https://aclanthology.org/2025.acl-long.382/) và code tại [edchengg/gollie-transfusion](https://github.com/edchengg/gollie-transfusion). Với cuộc thi, không chạy translation trong `run_inference.py` vì có thể phá offset, số liệu, tên thuốc và mã xét nghiệm.

Thêm `scripts/translate_annotated_corpus.py`:

```text
English clinical corpus
        ↓
protect entity/code/value bằng placeholder
        ↓
translate placeholder-preserving
        ↓
restore Vietnamese surface
        ↓
calculate raw offsets
        ↓
rule/manual validation
        ↓
train Vietnamese student
```

Phải bảo vệ `K21.9`, `360047`, `WBC`, `NEUT%`, `14,43`, tên thuốc và đơn vị. Nếu không được phép gọi dịch vụ ngoài, chạy bước này offline trước khi đóng gói dữ liệu/weights.

### 6. Character-level span ensemble

Ý tưởng character-level ensemble từ shared task multilingual clinical NER phù hợp với WER và boundary. Không cần dùng nguyên hệ thống MultiClinNER; chỉ thêm các model dự đoán xác suất bắt đầu/kết thúc trên raw character sequence.

Thêm `src/clinical_nlp/char_ensemble.py`:

```python
class CharacterSpanModel:
    def predict(self, raw_text: str) -> dict:
        # start_prob: [len(raw_text)]
        # end_prob: [len(raw_text)]
        # type_prob: [len(raw_text), 5]
        raise NotImplementedError

def ensemble_char_models(models, raw_text):
    outputs = [m.predict(raw_text) for m in models]
    start = mean([x["start_prob"] for x in outputs])
    end = mean([x["end_prob"] for x in outputs])
    typ = mean([x["type_prob"] for x in outputs])
    return decode_spans(start, end, typ, raw_text)
```

Decoder bắt buộc kiểm tra:

```text
start < end
span nằm trong raw text
type hợp lệ
không chứa heading/punctuation dư
không biến drug dose thành lab result
exact-align trước khi tạo Entity
```

Dùng out-of-fold predictions để chọn threshold riêng cho thuốc, triệu chứng, diagnosis, test name và result.

### 7. Assertions: 3 sigmoid heads + contextual rules

Assertions là multilabel. Thêm `src/clinical_nlp/assertion_model.py`:

```python
import torch
from torch import nn

class AssertionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 3)

    def forward(self, entity_context_hidden):
        # negated, family, historical
        return self.proj(entity_context_hidden)

def assertion_loss(logits, labels):
    return nn.BCEWithLogitsLoss()(logits, labels.float())
```

Input của head:

```text
left_context + entity_span + right_context
+ section_heading
+ list_prefix
+ nearest assertion cue
```

Kết hợp neural score với rule:

```python
def decide_assertion(neural_prob, rule_signal, threshold=0.5):
    if rule_signal == "hard_true":
        return True
    if rule_signal == "hard_false":
        return False
    return neural_prob >= threshold
```

Rule hard chỉ dành cho cue rõ như `không ghi nhận X`, `mẹ bệnh nhân bị X`, `thuốc trước nhập viện`. Không lan `isHistorical` từ section `Bệnh sử hiện tại`.

### 8. BioLORD/BERGAMOT/SapBERT: retrieval adapter

BioLORD-2023 và BERGAMOT được xây dựng quanh UMLS/biomedical concept representation; SapBERT có English và cross-lingual XLM-R variants. Chúng không tự nhiên biết bộ ICD/RxNorm của BTC và không chứng minh sẵn chất lượng tiếng Việt.

Nguồn:

- [BioLORD-2023](https://arxiv.org/abs/2311.16075)
- [BERGAMOT paper](https://aclanthology.org/2024.findings-naacl.288/) và [Andoree/BERGAMOT](https://github.com/Andoree/BERGAMOT)
- [SapBERT GitHub](https://github.com/cambridgeltl/sapbert)
- [xMEN cross-lingual normalization](https://github.com/hpi-dhc/xmen)
- [Glasgow-AI4BioMed/entitytools](https://github.com/Glasgow-AI4BioMed/entitytools)

Thêm `src/clinical_nlp/retrievers/base.py`:

```python
class ConceptRetriever:
    def encode_concepts(self, concepts):
        raise NotImplementedError

    def retrieve(self, mention, context, k=30):
        raise NotImplementedError

def retrieve_union(retrievers, mention, context, k=50):
    pools = [r.retrieve(mention, context, k) for r in retrievers]
    return unique_by_code(flatten(pools))
```

Index mỗi concept bằng nhiều view:

```text
preferred label
Vietnamese alias
English label
no-diacritic alias
abbreviation
parent/child description
drug ingredient/strength/form fields
```

Nếu dùng BERGAMOT đầy đủ, cần UMLS graph và training objective riêng; không trộn CUI vào output vì BTC yêu cầu ICD code/RxCUI.

### 9. Prompt-BioEL-style candidate reranking

Prompt-BioEL cho model nhìn nhiều candidate cùng lúc thay vì chấm từng candidate độc lập. Nguồn: [Prompt-BioEL GitHub](https://github.com/HITsz-TMG/Prompt-BioEL), [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/26624).

#### 9.1. Internal reranker input

```json
{
  "mention": "trào ngược dạ dày - thực quản",
  "context": "được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản",
  "type": "CHẨN_ĐOÁN",
  "candidates": [
    {"code":"K21.0","label":"GERD with esophagitis","parent":"K21"},
    {"code":"K21.9","label":"GERD without esophagitis","parent":"K21"}
  ]
}
```

Model trả score theo candidate hoặc index tốt nhất. Public output chỉ giữ code list.

#### 9.2. Fallback không train Prompt-BioEL

Rerank top 20–30 bằng cross-encoder nhỏ hoặc Qwen3-Reranker-0.6B, sau đó kết hợp:

```text
lexical_score
dense_score
reranker_score
parent_match
specificity evidence
ingredient/strength/form match
assertion compatibility
```

Không gọi reranker cho triệu chứng/test name vì chúng không có `candidates` trong schema.

### 10. ANGEL-style hard negatives

ANGEL dùng negative candidates và preference optimization cho biomedical entity linking. Với cuộc thi, nên lấy phần hard-negative construction trước vì nó trực tiếp phục vụ candidate Jaccard. Nguồn: [ANGEL paper](https://aclanthology.org/2025.findings-acl.558/), [dmis-lab/ANGEL](https://github.com/dmis-lab/ANGEL).

Thêm `scripts/build_hard_negatives.py`:

```python
def make_hard_negatives(gold_code, ontology, kind):
    if kind == "icd":
        return ontology.same_parent(
            gold_code,
            exclude=[gold_code],
            include_specificity_conflicts=True,
        )
    if kind == "rxnorm":
        return ontology.same_ingredient_different_strength_or_form(gold_code)
    return []
```

Nhóm negative cần có:

```text
ICD: same parent, acute/chronic, with/without complication, symptom/disease
RxNorm: ingredient-only, same ingredient khác strength, release/form khác
```

Training record:

```json
{"mention":"metoprolol succinate xl 50 mg",
 "context":"thuốc trước khi nhập viện",
 "positive":"866436",
 "negative":["866435","897656","197528"],
 "kind":"rxnorm"}
```

Hard-negative training không đồng nghĩa phải output tất cả candidate gần nhất. Candidate thứ hai chỉ được thêm khi vượt policy calibration.

### 11. GraphTREx-lite: graph nội bộ cho consistency

GraphTREx nghiên cứu span-based graph transformer cho temporal relation extraction trên clinical notes, nhưng output cuộc thi không có `relations`. Chỉ xây graph nhẹ sau khi có entities. Nguồn: [GraphTREx paper](https://aclanthology.org/2025.acl-long.1251/).

Thêm `src/clinical_nlp/concept_graph.py`:

```python
from dataclasses import dataclass

@dataclass
class Edge:
    source_id: int
    relation: str
    target_id: int | None
    score: float

def build_internal_graph(entities, doc):
    edges = []
    edges += connect_test_to_nearby_result(entities)
    edges += connect_drug_to_indication(entities, doc)
    edges += propagate_section_assertions(entities, doc)
    edges += connect_diagnosis_to_symptoms(entities, doc)
    return edges
```

Graph constraints có thể:

- truyền historical đến danh sách thuốc cùng section scope;
- ghép `WBC` với `14,43`;
- tránh coi liều thuốc là lab result;
- giảm ICD candidate không phù hợp type/context;
- phát hiện entity family nhưng lại gán assertion cho bệnh nhân.

Graph không được thêm vào output nếu schema BTC không yêu cầu.

### 12. Cấu hình model an toàn theo giới hạn 9B

#### Profile R: khuyến nghị cho inference ổn định

```text
NER student: XLM-R/PhoBERT/ViPubmedDeBERTa 100–400M
Assertion: dùng chung encoder + 3 sigmoid heads
Embedding: Qwen3-Embedding-0.6B hoặc SapBERT-XLMR
Reranker: Qwen3-Reranker-0.6B hoặc cross-encoder nhỏ
Graph: Python rules nhẹ
```

Các model nhỏ chạy runtime; Qwen3-8B chỉ chạy offline để sinh/kiểm dữ liệu.

#### Profile L: ưu tiên LLM fallback

```text
Qwen3-4B/8B: chỉ xử lý span mơ hồ hoặc adjudication
Embedding/reranker: chạy tuần tự và benchmark VRAM
Rules: luôn chạy trước LLM
```

Không mặc định chạy `Qwen3-8B + Embedding-0.6B + Reranker-0.6B` đồng thời nếu BTC tính tổng tham số. Profile R an toàn hơn cho source reconstruction và timeout.

### 13. Lệnh tích hợp end-to-end

```bash
# 0. Kiểm tra repo/model/index đã pin
python scripts/doctor.py --config configs/inference.yaml

# 1. Build ontology views, alias table, lexical index và vectors tùy chọn
python scripts/build_ontology_index.py \
  --icd data/ontology/icd/source.json \
  --rxnorm data/ontology/rxnorm/RXNCONSO.RRF \
  --retriever lexical,dense \
  --output artifacts/ontology_index

# 2. Tạo structured records và synthetic text
python scripts/generate_guidex_data.py \
  --ontology artifacts/ontology_index \
  --n-records 30000 \
  --styles discharge_note_vi,doctor_note_vi,lab_report_vi,medication_list \
  --output data/synthetic/guidex

# 3. Optional: annotation projection có bảo vệ placeholder
python scripts/translate_annotated_corpus.py \
  --input data/external/clinical_en \
  --output data/synthetic/transfusion_vi \
  --protect-codes --protect-values

# 4. Sinh hard negatives
python scripts/build_hard_negatives.py \
  --ontology artifacts/ontology_index \
  --records data/synthetic/guidex/accepted.jsonl \
  --output data/linking/hard_negatives.jsonl

# 5. Train student nếu ablation cho thấy có lợi
python scripts/train_span_model.py --config configs/train_span.yaml
python scripts/train_assertion_model.py --config configs/train_assertion.yaml
python scripts/train_reranker.py --config configs/train_reranker.yaml

# 6. Inference
python scripts/run_inference.py \
  --input_dir test/input --output_dir output \
  --config configs/inference.yaml \
  --enable-character-ensemble \
  --enable-hard-negative-rerank

# 7. Strict validation và package
python scripts/validate_output.py \
  --input_dir test/input --output_dir output --strict
python scripts/package_submission.py \
  --output_dir output --zip_path output.zip
```

### 14. Thứ tự triển khai và tiêu chí giữ/bỏ

| Sprint | Module | Tiêu chí giữ |
|---|---|---|
| S0 | raw span, schema, validator | 100/100 file JSON hợp lệ; text khớp raw slice |
| S1 | rules + dictionary + baseline encoder | recall tăng mà không làm span dư nhiều |
| S2 | GUIDEX synthetic + student NER | exact boundary/type tăng trên manual-dev |
| S3 | dense retrieval + hard negatives | candidate recall/Jaccard tăng |
| S4 | listwise reranker | candidate Jaccard tăng sau calibration |
| S5 | multilabel assertion | assertion Jaccard tăng, historical precision không giảm |
| S6 | graph-lite | giảm lỗi test-result, scope và ontology inconsistency |
| S7 | LLM fallback | điểm tăng sau khi trừ latency/VRAM |

Mỗi module phải có ablation riêng. Không giữ module chỉ vì paper báo cáo state-of-the-art trên dataset khác.

### 15. Các nghiên cứu không nên tích hợp nguyên bản

- **OpenBioNER-v2**: language/boundary/nested-span mismatch; dùng teacher hoặc fine-tune lại.
- **MedCAT v2**: model pack và ontology khác; dùng registry/MetaCAT design, không dùng output pack trực tiếp.
- **BioLORD/BERGAMOT**: cần kiểm tra language và KB mismatch; dùng retriever/feature trước full retraining.
- **TransFusion**: chỉ offline vì translation runtime có thể phá offset.
- **GraphTREx**: relation/temporal task khác output; chỉ dùng graph-lite.
- **ANGEL**: lấy hard negatives trước preference optimization.
- **Prompt-BioEL**: chỉ rerank top-k nhỏ; không đưa hàng nghìn candidate vào prompt.

Kết luận thực dụng: research proposal nên được tích hợp như một **research-enhanced modular pipeline**, không thay toàn bộ pipeline deterministic bằng một model duy nhất.


| Nội dung bản cũ | Quyết định |
|---|---|
| Không dùng ViPubmedDeBERTa/XLM-R trực tiếp như NER | Giữ nguyên |
| Exact-span alignment, không để LLM sinh offset | Giữ nguyên |
| Dictionary + regex + section rules | Giữ nhưng mở rộng result chữ và scope |
| Qwen3-8B làm semantic extractor | Giữ như một lựa chọn, cần benchmark/pin revision |
| Hai lượt extraction cho toàn bộ tài liệu | Bỏ mặc định; chỉ verify ca khó |
| `top-1` ICD/RxNorm mặc định | Bỏ; thay bằng calibrated dynamic top-k |
| `Tiền sử bệnh` lan historical rộng | Bỏ; thêm current-section override và scope boundary |
| Lab result chỉ nhận số | Bỏ; thêm positive/negative/textual result |
| Fine-tune là tùy chọn hoàn toàn | Sửa thành nhánh submission nên có weak/synthetic data và calibration |
| Qwen3-8B + 0.6B embedding + 0.6B reranker | Chỉ dùng nếu BTC xác nhận cap theo từng model; nếu không, chọn phương án A hoặc B |
| Tải ontology mới nhất tùy ý | Bỏ; pin đúng release/KB mà BTC dùng |

### 16. ViPubMedDeBERTa-based upgrade: encoder trước, student NER sau

ViPubMedDeBERTa là pretrained Vietnamese biomedical encoder. Nó không thay thế
Qwen trong vai trò chat extractor nếu chưa có token-classification head; model
được dùng qua `AutoModel`/`AutoModelForTokenClassification`, không qua
`llama-server` chat completion. Vì vậy dev team cần triển khai theo hai profile
độc lập và có ablation riêng.

#### 16.1. Profile V1: semantic reranker không fine-tune

Đây là profile có thể tích hợp trước khi tạo dữ liệu huấn luyện:

```text
raw document
    ↓
DocumentView + section/line split
    ↓
Qwen extractor hoặc rule/LLM candidate extraction
    ↓
raw character alignment + span/type resolver
    ↓
deterministic assertion resolver
    ↓
ICD/RxNorm exact → normalized → fuzzy retrieval
    ↓
ViPubMedDeBERTa encoder reranker trên top-k nhỏ
    ↓
score/margin + attribute constraints
    ↓
final link_entities()
    ↓
public JSON
```

ViPubMedDeBERTa chỉ nhìn các candidate đã được lexical linker thu hẹp, không
encode toàn bộ ontology ở mỗi request. Candidate input nội bộ gồm:

```json
{
  "mention": "viêm túi mật cấp",
  "context": "siêu âm gợi ý viêm túi mật cấp",
  "type": "CHẨN_ĐOÁN",
  "candidate": {"code": "K81.0", "label": "Acute cholecystitis"}
}
```

Thêm các interface:

```python
class ViPubMedEncoder:
    def encode(self, texts: list[str]) -> "Array":
        """Return normalized pooled hidden states; no public offsets."""
        raise NotImplementedError

class SemanticCandidateReranker:
    def rerank(self, mention, context, candidates, entity_type):
        """Return candidates with internal semantic_score and margin."""
        raise NotImplementedError
```

Đặt reranker sau `Ontology.lookup_scored()` và trước `link_entities()` final.
Không dùng semantic score để ghi đè các hard constraints:

```text
route/form/strength conflict → reject
wrong entity type → reject
procedure/heading/test result → reject
low lexical score → no candidate
small semantic margin → keep controlled set or empty
```

Để model nhận tiếng Việt đúng cách, có thể word-segment riêng cho chuỗi đưa
vào encoder bằng PyVi. Chuỗi segment này chỉ là model view; `text` và
`position` public luôn lấy từ raw document, tuyệt đối không lấy offset sau
segmentation.

Không nên coi cosine similarity từ pretrained MLM là calibrated linker score.
V1 phải lưu `lexical_score`, `semantic_score`, `margin`, `model_revision` và
`match_mode` vào audit log; chỉ bật semantic reranking công khai sau khi có
manual-dev calibration.

#### 16.2. Profile V2: ViPubMedDeBERTa làm NER student

Sau khi có dữ liệu synthetic/weak-label đã được validator và người kiểm tra
duyệt, fine-tune:

```text
ViPubMedDeBERTa backbone
    + AutoModelForTokenClassification
    + BIO/type labels
    → character span reconstruction
    → boundary/type resolver
    → assertion resolver
    → ICD/RxNorm linker
```

Label set bắt buộc:

```text
O
B/I-TRIỆU_CHỨNG
B/I-TÊN_XÉT_NGHIỆM
B/I-KẾT_QUẢ_XÉT_NGHIỆM
B/I-CHẨN_ĐOÁN
B/I-THUỐC
```

Thêm `vipubmed_ner.py` với contract:

```python
@dataclass
class TokenSpanCandidate:
    start: int
    end: int
    text: str
    entity_type: str
    score: float
    source: str = "vipubmed_ner"

class ViPubMedNER:
    def predict(self, raw_text: str) -> list[TokenSpanCandidate]:
        """Decode BIO labels, then align spans to raw_text."""
        raise NotImplementedError
```

`ViPubMedNER` chỉ thay `llm_extract()`/`align_llm()` ở profile V2. Các tầng
`resolve`, assertion, ontology linker, candidate policy và public schema vẫn
được dùng chung. Như vậy team có thể benchmark Qwen extractor và ViPubMed NER
trên cùng một downstream pipeline.

#### 16.3. Model manifest và cache lifecycle

Mỗi checkpoint phải có manifest riêng, không tải model ngầm trong inference:

```json
{
  "model_id": "manhtt-079/vipubmed-deberta-base",
  "revision": "<pinned-revision>",
  "sha256": "<artifact-sha256>",
  "task": "reranker|token_classification",
  "max_length": 512,
  "tokenizer_revision": "<pinned-revision>",
  "device_profile": "cuda_sm52|cpu",
  "training_data_hash": null,
  "calibration_version": "v0"
}
```

Concept vectors phải được build offline theo:

```text
ontology snapshot hash
+ derived index hash
+ alias table hash
+ model revision
+ pooling/config hash
    → vector cache key
```

Nếu một hash thay đổi, rebuild vector cache; không trộn vector của các release
ontology/model khác nhau. Inference chỉ đọc cache và fail-closed nếu manifest
không khớp.

#### 16.4. Cấu hình runtime

Thêm profile thay vì thay đổi ngầm model hiện tại:

```yaml
model_profile: qwen_extractor_v1
vipubmed:
  enabled: false
  mode: reranker              # reranker | token_classification
  model_id: manhtt-079/vipubmed-deberta-base
  revision: <pinned-revision>
  device: cuda
  max_length: 512
  batch_size: 8
  lexical_top_k: 20
  semantic_top_k: 3
  min_score: 0.78
  min_margin: 0.05
  vector_cache: artifacts/vipubmed_vectors
```

Profile vận hành:

```text
QWEN_ONLY       → extractor hiện tại, linker deterministic
QWEN_VIPUBMED   → Qwen + ViPubMed reranker trên top-k
VIPUBMED_NER    → fine-tuned ViPubMed thay extractor
```

Không chạy đồng thời Qwen 8B, embedding model và reranker nếu server bị giới
hạn VRAM hoặc tổng tham số. Quadro M6000 24GB có thể chạy encoder nhỏ, nhưng
phải benchmark riêng với đúng PyTorch/CUDA build và không khởi động hai server
chiếm cùng GPU nếu chưa đo memory peak.

#### 16.5. Thứ tự triển khai cho dev team

| Phase | Deliverable | Điều kiện giữ |
|---|---|---|
| V0 | `ViPubMedEncoder` + model manifest + smoke test | load model offline, checksum/revision hợp lệ |
| V1 | lexical top-k + semantic reranker adapter | không làm giảm candidate precision trên manual-dev |
| V2 | vector cache theo ontology/model hash | reproducible, rebuild khi hash đổi |
| V3 | calibration semantic score/margin | candidate Jaccard tăng, no-hit không giảm an toàn |
| V4 | BIO fine-tuning student NER | exact span/type vượt Qwen baseline |
| V5 | assertion head dùng chung encoder | assertion Jaccard tăng, historical precision không giảm |
| V6 | production profile/rollback | đổi model bằng config, không sửa public schema |

Benchmark bắt buộc cho mỗi phase:

```text
span exactness + WER/Jaccard
type accuracy và type-confusion matrix
assertion Jaccard theo file/type
candidate coverage, candidate Jaccard, no-hit rate
latency p50/p95, peak VRAM, throughput
ablation: QWEN_ONLY vs QWEN_VIPUBMED vs VIPUBMED_NER
```

Quyết định production mặc định: giữ Qwen làm extractor cho đến khi
`VIPUBMED_NER` có manual-dev evidence tốt hơn; bật ViPubMed reranker chỉ sau
calibration và luôn có cờ rollback về `QWEN_ONLY`.
