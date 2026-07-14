# Kiến trúc pipeline khuyến nghị cho cuộc thi chuẩn hóa văn bản y khoa

## Kết luận đánh giá

Tài liệu kiến trúc ban đầu **đúng về hướng tổng thể**, đặc biệt ở các điểm:

- không dùng trực tiếp backbone pretrained như một NER đã được huấn luyện;
- tách extraction, assertion, linking và offset alignment;
- lấy `text` từ substring nguyên bản thay vì để LLM tự sửa văn bản;
- dùng dictionary/regex để tăng recall thuốc và xét nghiệm;
- chuẩn bị pipeline offline để chạy được trên private test.

Tuy nhiên, tài liệu **chưa phải thiết kế sẵn sàng triển khai**. Có các lỗi/rủi ro sau:

1. Gán `isHistorical` theo section/từ khóa quá rộng, dễ nhầm “Tiền sử bệnh hiện tại” với tiền sử thật.
2. Lab parser thiên về giá trị số, trong khi dữ liệu mẫu có kết quả dạng chữ, tính từ, âm tính/dương tính và kết quả hình ảnh.
3. `top-1` ICD/RxNorm được đặt làm mặc định nhưng không được suy ra từ metric; ví dụ chính thức có chẩn đoán với nhiều mã ICD.
4. Qwen3-8B + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B có thể vượt giới hạn 9B nếu BTC tính tổng số tham số; đồng thời hai lượt LLM toàn văn có nguy cơ vượt thời gian 600 giây.
5. Việc tạo dữ liệu để huấn luyện bị mô tả như tùy chọn, trong khi đề bài nêu thí sinh cần dùng giải pháp bổ sung để tạo dữ liệu huấn luyện.
6. Chưa có cơ chế calibration cho span, type, assertion và số lượng candidate khi không có nhãn công khai.
7. Chưa khóa version/hash của ontology, model, tokenizer, quantization và quy tắc output.

Vì vậy, nên dùng thiết kế dưới đây làm bản triển khai chính; bản cũ chỉ nên xem là baseline ý tưởng.

## 1. Bằng chứng từ bộ input đã kiểm tra

Đã kiểm tra `input.zip` gồm 100 file `.txt`, tổng khoảng 132.336 ký tự. Bộ mẫu có cấu trúc bệnh án bán cấu trúc nhưng không đồng nhất:

| Đặc điểm quan sát | Số file có dấu hiệu | Hệ quả thiết kế |
|---|---:|---|
| Section bệnh sử/đánh giá/xét nghiệm | 100/100 | Dùng section làm feature, không dùng làm ground truth |
| Cue thời gian/tiền sử | 94/100 | Cần temporal scope, không gán historical chỉ bằng regex |
| Cue phủ định/chưa/âm tính/không | 86/100 | Cần negation scope theo câu và danh sách |
| Cue gia đình/người nhà | 9/100 | Cần rule family riêng, tránh gán nhầm cho bệnh nhân |
| Từ khóa xét nghiệm/chẩn đoán hình ảnh | 73/100 | Tách lab result khỏi modality/hình ảnh |
| Thuốc, liều, đường dùng hoặc tần suất | 37/100 | Drug matcher không được phụ thuộc vào việc có strength |

Các file có những dạng cần xử lý riêng, ví dụ:

- `Tiền sử bệnh` và `Bệnh sử hiện tại` cùng xuất hiện trong một hồ sơ;
- thuốc dạng `metoprolol 25mg po bid`, nhưng cũng có thuốc thiếu strength như `guaifenesin ml po q6h:prn`;
- kết quả như `CEA tăng nhẹ lên 4.9`, `canxi là 12.0`, `âm tính`, `không có bệnh động mạch vành`;
- viết tắt và code-switching: `MRI`, `FM +`, `VB -`, `LOF -`, `WBC`, `TURP`;
- lỗi dính từ, thiếu khoảng trắng và lỗi chính tả/đánh máy;
- một số dòng không có dấu câu hoặc nhiều section nằm trên cùng một dòng.

Các thống kê trên là thống kê từ input, **không phải nhãn ground truth**. Không được dùng chúng để suy ra nhãn cụ thể.

## 2. Hợp đồng dữ liệu bắt buộc

### 2.1. Input và output

Với mỗi `input/{n}.txt`, sinh đúng `output/{n}.json`. Output là một JSON array; không bọc thêm object cấp cao.

Khi nộp, archive phải có dạng:

```text
output.zip
└── output/
    ├── 1.json
    ├── 2.json
    ├── ...
    └── 100.json
```

Phải sinh đủ file, kể cả file có kết quả `[]`. JSON dùng UTF-8, không BOM, không thêm markdown fence.

### 2.2. Schema theo type

Chỉ dùng đúng năm type:

```text
TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
CHẨN_ĐOÁN
THUỐC
```

Khung tối thiểu:

```json
{
  "text": "substring nguyên bản",
  "type": "THUỐC",
  "position": [58, 83]
}
```

Với `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, `THUỐC`, thêm:

```json
"assertions": []
```

Với `CHẨN_ĐOÁN`, `THUỐC`, thêm:

```json
"candidates": []
```

Không thêm `assertions` hoặc `candidates` cho `TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM` nếu output mẫu chính thức không có các field đó. Đây là điểm phải kiểm tra lại bằng validator của BTC nếu họ cung cấp validator.

### 2.3. Position

Ví dụ chính thức phù hợp với quy ước Python `[start, end)`:

```python
assert original_text[start:end] == entity["text"]
```

Không tính position trên text đã lower-case, bỏ dấu, chuẩn hóa khoảng trắng hoặc thêm line marker. Nếu LLM trả text đã chuẩn hóa, phải align ngược về một occurrence trong raw text; nếu không align được thì loại candidate đó.

Mỗi occurrence là một entity riêng. Chỉ deduplicate khi `start`, `end` và type trùng nhau; không deduplicate theo nội dung `text`.

## 3. Kiến trúc inference được khuyến nghị

```text
Raw UTF-8 text
    │
    ├─ lưu line/sentence/offset nguyên bản
    ├─ phát hiện section và temporal scope
    │
    ├─ deterministic candidate generation
    │    ├─ RxNorm drug dictionary + medication parser
    │    ├─ lab/test dictionary + result grammar
    │    ├─ symptom/diagnosis alias dictionary
    │    └─ typo/spacing tolerant search
    │
    ├─ local LLM semantic extraction cho span/type còn thiếu hoặc mơ hồ
    │
    ├─ exact-span alignment và overlap resolution
    │
    ├─ assertion resolver: negation + family + historical
    │
    ├─ ICD/RxNorm candidate retrieval và reranking offline
    │
    ├─ candidate-count calibration và schema validation
    │
    └─ 1.json ... 100.json → output.zip
```

Điểm quan trọng là LLM không nên là nguồn phát hiện duy nhất. Với các pattern chắc chắn, rule/dictionary nhanh hơn, ổn định hơn và không tốn lượt sinh token.

## 4. Tiền xử lý và span alignment

Không thay đổi `original_text`. Tạo metadata song song:

```python
Line(
    line_id="L002",
    raw_start=...,       # offset trong original_text
    raw_end=...,         # exclusive
    raw_text=...,        # nguyên bản
    normalized_text=...  # chỉ dùng cho search
)
```

Chuẩn hóa phục vụ tìm kiếm có thể gồm:

- Unicode NFC;
- lower-case bản sao;
- chuẩn hóa khoảng trắng bản sao;
- bỏ dấu bản sao cho fuzzy retrieval;
- biến thể dấu câu;
- mapping lỗi dính từ.

Không được dùng normalized text để ghi vào `text` hoặc tính `position`.

LLM chỉ trả về:

```json
{
  "line_id": "L002",
  "text": "metoprolol 25mg po bid",
  "type": "THUỐC",
  "assertions": []
}
```

Aligner thực hiện theo thứ tự:

1. exact substring trong line;
2. exact match sau khi bỏ khác biệt khoảng trắng nhưng trả lại span raw;
3. fuzzy token alignment cho lỗi dính/typo;
4. nếu có nhiều occurrence, chọn occurrence gần context do LLM cung cấp;
5. nếu điểm align dưới threshold hoặc span không chắc chắn, bỏ entity thay vì sinh text không tồn tại.

Mọi entity trước khi serialize phải qua:

```python
assert entity.text == original_text[entity.start:entity.end]
assert 0 <= entity.start < entity.end <= len(original_text)
```

## 5. Nhận diện entity và type

### 5.1. Thuốc

Drug matcher phải nhận được cả:

- tên hoạt chất, brand name, clinical drug name;
- strength có hoặc không có;
- dạng bào chế, release type;
- route/frequency: `po`, `oral`, `iv`, `bid`, `qid`, `prn`, `daily`;
- lỗi khoảng trắng, dấu gạch, viết hoa/thường và tiếng Anh trong câu tiếng Việt.

Không bắt buộc phải có strength. `guaifenesin ml po q6h:prn` là ví dụ cho thấy regex chỉ tìm số + đơn vị sẽ bỏ sót thuốc.

Span thuốc nên bao gồm phần biểu đạt thuốc và thuộc tính dùng thuốc mà gold sample thể hiện, nhưng không kéo theo câu điều trị phía sau. Dùng parser để tách ingredient/strength/form/route/frequency nội bộ; chỉ output một span theo format của đề.

### 5.2. Tên xét nghiệm

Dùng dictionary và cue theo section, bao gồm:

- tên xét nghiệm/lab marker: `WBC`, `CEA`, `canxi`, `troponin`;
- viết tắt và tên có diễn giải trong ngoặc;
- thành phần công thức xét nghiệm như `NEUT%`, `LYPH%` nếu xuất hiện trong mẫu.

Không coi mọi `MRI`, `siêu âm`, `chụp`, `sinh thiết` là cùng một type nếu guideline của BTC chỉ định chúng là chẩn đoán hình ảnh/thủ thuật. Cần cấu hình riêng `imaging/procedure` để tránh đẩy mọi modality vào `TÊN_XÉT_NGHIỆM`.

### 5.3. Kết quả xét nghiệm

Không giới hạn result ở số thập phân. Result grammar cần bao phủ:

```text
TEST: 14,43
TEST là 12.0
TEST tăng nhẹ lên 4.9
TEST âm tính
TEST dương tính
TEST không phát hiện ...
```

Chỉ gán `KẾT_QUẢ_XÉT_NGHIỆM` khi có test/measurement cue trong cùng câu, cùng dòng hoặc cửa sổ context đã cấu hình. Không bắt nhầm:

- tuổi, ngày tháng, thời gian khởi phát;
- liều thuốc;
- huyết áp, nhịp tim nếu guideline không xếp chúng là xét nghiệm;
- kích thước tổn thương;
- số lần ngất hoặc thời lượng triệu chứng.

Đặc tả mô tả result gồm giá trị và đơn vị, nhưng ví dụ chính thức có trường hợp output chỉ là giá trị. Vì vậy parser phải giữ đúng text span theo quy tắc gold: nếu đơn vị nằm trong span mẫu thì giữ đơn vị; nếu không có đơn vị thì không tự thêm.

### 5.4. Triệu chứng và chẩn đoán

LLM hữu ích để phân biệt:

- triệu chứng bệnh nhân khai báo;
- chẩn đoán bác sĩ hoặc bệnh đã được xác định;
- bệnh chỉ nằm trong tiền sử;
- cùng một cụm từ xuất hiện trong câu phủ định hoặc của người nhà.

Section heading như `Triệu chứng hiện tại`, `Các bệnh lý mãn tính`, `Kết quả xét nghiệm` không tự thân là entity. Lý do nhập viện và các bullet bên dưới phải được phân tích theo ngữ cảnh, không dùng heading làm nhãn cứng.

## 6. Assertion resolver

`assertions` chỉ áp dụng cho `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, `THUỐC` và phải là tập con của:

```text
isNegated
isFamily
isHistorical
```

Một entity có thể có nhiều assertion; ví dụ bệnh của người nhà được phủ định có thể cần cả `isFamily` và `isNegated` nếu guideline xác định như vậy.

### 6.1. Historical

Dùng ba lớp bằng chứng, ưu tiên từ trên xuống:

1. cue cục bộ: `tiền sử ...`, `đã từng`, `trước đây`, `đã điều trị`, `thuốc trước khi nhập viện`;
2. section scope: danh sách dưới `Thuốc trước khi nhập viện`, `Tiền sử phẫu thuật`;
3. LLM adjudication khi câu vừa có tiền sử vừa mô tả hiện tại.

Các rule phải có override:

- `Bệnh sử hiện tại`, `Triệu chứng hiện tại`, `Lý do nhập viện` mặc định không historical;
- không lan historical qua section mới;
- không lan historical từ heading sang toàn bộ văn bản nếu không có boundary danh sách;
- “trước khi nhập viện” không tự động làm mọi triệu chứng trong câu historical nếu câu đang mô tả lý do nhập viện hiện tại.

Đây là điểm bản cũ cần sửa mạnh nhất.

### 6.2. Negation

Xây cue và scope cho:

```text
không, không có, không ghi nhận, phủ nhận, chưa, âm tính,
không thấy, không phát hiện, không bằng chứng của
```

Resolver phải xử lý coordination/list:

```text
không buồn nôn, nôn hoặc đổ mồ hôi
```

Các entity trong list có thể cùng nhận `isNegated`. Cue chỉ áp dụng trong cửa sổ scope, không áp dụng cho câu kế tiếp nếu không có section/list continuation.

### 6.3. Family

Chỉ bật `isFamily` khi entity nằm trong quan hệ với `mẹ`, `bố/cha`, `anh chị em`, `người nhà`, `gia đình`, `họ hàng` hoặc đại từ tương đương. Không gán family chỉ vì câu có từ `gia đình`; cần xác định bệnh/symptom đang thuộc người khác.

## 7. ICD-10 và RxNorm linking

### 7.1. Nguyên tắc nguồn dữ liệu

Không gọi API online trong inference. Tạo index offline từ nguồn/knowledge base mà BTC cho phép và khóa:

```text
ontology_name
release/version
download_date
source file checksum
normalization rules
```

RxNorm được phát hành theo các bản monthly/weekly; RXCUI và tên có thể thay đổi trạng thái giữa các release. Không được dùng một release tải ngẫu nhiên rồi giả định luôn khớp với candidate ground truth. Nếu BTC cung cấp KB, KB của BTC là nguồn ưu tiên.

ICD cũng phải xác định rõ là ICD-10 WHO, ICD-10-CM hay một danh mục dẫn xuất. Không trộn mã, mô tả và parent hierarchy từ các version khác nhau.

### 7.2. ICD candidate generation

Tạo index theo:

- code;
- preferred label;
- synonym/alias;
- parent/child code;
- mô tả tiếng Anh và tiếng Việt nếu có.

Sinh candidate từ nhiều kênh:

```text
exact alias/code match
normalized lexical/BM25
fuzzy token match cho typo
dense retrieval trên mention + context
ontology parent/child expansion
```

Không đưa candidate chỉ vì embedding gần nếu concept thuộc nhóm sai rõ ràng. Reranker phải nhìn cả mention, câu, section và mã/mô tả candidate.

### 7.3. RxNorm candidate generation

Parser tách nội bộ:

```json
{
  "ingredient": "metoprolol succinate",
  "strength": "50 mg",
  "dose_form": null,
  "release": "extended release",
  "route": "oral",
  "frequency": "daily"
}
```

Retrieval ưu tiên:

```text
ingredient + strength + dose form + release
ingredient + strength + dose form
ingredient + strength
ingredient + release
ingredient only
```

`daily`, `bid`, `qid`, `prn` thường là hướng dẫn dùng, không dùng chúng để phân biệt RxCUI nếu chúng không nằm trong normalized drug concept.

### 7.4. Số lượng candidate

Không đặt `top-1` làm luật cứng. Metric dùng Jaccard nên:

- candidate thiếu bị phạt recall;
- candidate thừa bị phạt precision;
- nếu gold có hai mã tương đương/khác mức chi tiết, top-1 chắc chắn không đủ.

Dùng candidate policy có calibration:

```text
exact unambiguous alias → 1 candidate
exact drug product match → 1 candidate
ambiguous ingredient/strength → 2–3 candidate
ICD có nhiều mức specificity → giữ các candidate cùng nhánh nếu margin nhỏ
```

Lưu score/margin nội bộ, nhưng output chỉ chứa mã. Không output top-5 mặc định. Threshold và max-k phải được quyết định trên tập calibration có gán nhãn, không phải bằng trực giác.

## 8. Không có nhãn công khai: cách tạo dữ liệu và calibration

### 8.1. Tập calibration thủ công

Gán nhãn 20–40 file đại diện, phân tầng theo:

- bệnh sử dài/ngắn;
- thuốc nhiều/ít;
- lab và imaging;
- negation/family/history;
- lỗi dính từ và viết tắt.

Chia thành train/dev hoặc dùng leave-one-document-out. Không báo cáo điểm trên dữ liệu đã dùng để điều chỉnh rule như thể đó là điểm private.

Nhãn cần gồm span, type, assertion, candidate và lý do boundary. Đây là nguồn quan trọng hơn việc chỉ hỏi LLM “hãy trích xuất tất cả”.

### 8.2. Weak labels

Sinh nhãn từ:

- RxNorm/ICD exact alias;
- lab/test dictionary;
- medication/result grammar;
- section/cue rules;
- LLM consensus với hai prompt độc lập.

Chỉ dùng nhãn có confidence cao cho student model. Mỗi weak label cần lưu provenance để loại bỏ khi rule đó bị phát hiện sai.

### 8.3. Synthetic data

Sinh văn bản theo template nhưng phải có biến thể:

- tiếng Việt, tiếng Anh, code-switching;
- viết tắt, lỗi dính từ, lỗi dấu;
- section và bullet;
- phủ định, family, history;
- result số và result chữ;
- thuốc có/không có strength.

Dùng marker tạm thời để tạo span:

```text
<E1 type="TRIỆU_CHỨNG">đau thượng vị</E1>
```

Sau đó xóa marker và tính offset bằng code. Không dùng offset do model sinh.

Synthetic data chỉ được đưa vào train sau khi kiểm tra một mẫu bằng người hoặc rule verifier; không để một LLM tự sinh và tự chấm toàn bộ nhãn.

## 9. Model và ngân sách tính toán

Qwen3-8B là một lựa chọn hợp lý cho local semantic extraction trong giới hạn từng model dưới 9B, nhưng phải pin revision và chạy non-thinking mode cho extraction JSON có cấu trúc. Thinking mode chỉ nên dùng cho ít ca mơ hồ vì tốn thời gian và dễ sinh output không cần thiết.

Hai phương án an toàn:

### Phương án A: ưu tiên chất lượng extraction

```text
Qwen3-8B quantized
+ BM25/FAISS static retrieval
+ rules + exact dictionary
```

Không chạy thêm embedding/reranker model đồng thời nếu BTC tính tổng tham số.

### Phương án B: ưu tiên tổng ngân sách model

```text
Qwen3-4B hoặc model local nhỏ hơn
+ Qwen3-Embedding-0.6B
+ Qwen3-Reranker-0.6B tùy chọn
+ rules + BM25
```

Tổng model, VRAM/RAM, quantization và thời gian phải được benchmark trên máy chấm giả lập. Không coi `8B + 0.6B + 0.6B = 8B` là an toàn; nếu giới hạn áp dụng trên tổng số tham số thì tổ hợp này vượt 9B.

Không chạy hai lượt LLM toàn văn cho cả 100 file theo mặc định. Thay vào đó:

1. rule/dictionary pass;
2. một LLM pass cho document hoặc chunk;
3. verification pass chỉ cho entity không chắc chắn, overlap hoặc candidate margin thấp;
4. cache prompt/output và precompute ontology index trước khi inference.

Đặt giới hạn `max_new_tokens`, batch hợp lý và timeout mỗi document. Ghi lại runtime theo từng stage để bảo đảm dưới 600 giây.

## 10. Pipeline triển khai cụ thể

```python
def process_document(original_text: str) -> list[dict]:
    doc = build_document_view(original_text)
    sections = detect_sections(doc)

    candidates = []
    candidates += match_rxnorm_drugs(doc)
    candidates += match_labs_and_results(doc)
    candidates += match_symptom_diagnosis_aliases(doc)

    llm_items = llm_extract_only_missing_or_ambiguous(
        numbered_raw_lines=doc.lines,
        schema=official_schema,
        examples=official_examples,
        thinking=False,
    )
    candidates += align_llm_items(llm_items, doc)

    entities = resolve_overlaps_and_same_occurrence(candidates)

    for entity in entities:
        if entity.type in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}:
            entity.assertions = resolve_assertions(
                entity, doc, sections, rules, optional_llm=True
            )

        if entity.type == "CHẨN_ĐOÁN":
            pool = retrieve_icd(entity.text, entity.context, ontology_index)
            entity.candidates = select_calibrated_candidates(pool)

        if entity.type == "THUỐC":
            attrs = parse_medication(entity.text)
            pool = retrieve_rxnorm(attrs, entity.context, ontology_index)
            entity.candidates = select_calibrated_candidates(pool)

    entities = validate_and_prune(entities, original_text)
    return serialize_by_official_schema(entities)
```

Trước khi tạo ZIP, chạy validator:

```python
for n in range(1, 101):
    assert output_path(f"{n}.json").exists()
    data = json.load(open(output_path(f"{n}.json"), encoding="utf-8"))
    assert isinstance(data, list)
    for e in data:
        assert e["text"] == raw[n][e["position"][0]:e["position"][1]]
        assert e["type"] in ALLOWED_TYPES
        assert e["position"][0] < e["position"][1]
        if e["type"] in ASSERTION_TYPES:
            assert set(e.get("assertions", [])) <= ALLOWED_ASSERTIONS
        if e["type"] in {"CHẨN_ĐOÁN", "THUỐC"}:
            assert all(isinstance(x, str) for x in e.get("candidates", []))
```

## Phần triển khai chi tiết end-to-end

Phần này bổ sung trực tiếp các nội dung triển khai còn thiếu trong file pipeline gốc: source code, cấu trúc thư mục, model, cách build ontology, interface giữa các module, lệnh chạy và kiểm thử. Các đoạn code là skeleton có thể dùng để bắt đầu; cần nối với implementation cụ thể của đội và validator chính thức của BTC.

### A. Bộ công cụ và source code tham chiếu

| Thành phần | Dùng để làm gì | Source chính thức/tham khảo |
|---|---|---|
| Qwen3-8B | extraction semantic, type, adjudication | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) |
| Transformers | load LLM/encoder, tokenizer | [huggingface/transformers](https://github.com/huggingface/transformers) |
| vLLM | local serving/batching Qwen trên GPU | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| Qwen3 Embedding/Reranker | dense retrieval và reranking | [QwenLM/Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) |
| FAISS | vector index offline | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| Sentence Transformers | embedding/evaluation tiện dụng | [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) |
| RapidFuzz | fuzzy match tên thuốc/test | [maxbachmann/RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) |
| medspaCy | khung section/ConText tham khảo | [medspacy/medspacy](https://github.com/medspacy/medspacy) |
| ViPubmedDeBERTa | encoder NER student sau khi có nhãn | [manhtt-079/vipubmed-deberta](https://github.com/manhtt-079/vipubmed-deberta) |
| XLM-RoBERTa | assertion/type student sau khi có nhãn | [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) |
| MedXN | thiết kế parser medication tham khảo | [OHNLP/MedXN](https://github.com/OHNLP/MedXN) |
| SapBERT | biomedical linking baseline tùy chọn | [cambridgeltl/sapbert](https://github.com/cambridgeltl/sapbert) |
| RxNorm | dữ liệu RxCUI, không gọi API runtime | [NLM RxNorm files](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html) |

Không cài tất cả thành phần ngay từ đầu. Baseline nên bắt đầu bằng `transformers + regex + rapidfuzz + BM25/FAISS`; chỉ thêm embedding/reranker sau khi đã đo được lỗi linking trên calibration set.

### B. Cấu trúc repository đề xuất

```text
medical-nlp-contest/
├── README.md
├── pyproject.toml
├── requirements.lock.txt
├── configs/
│   ├── inference.yaml
│   ├── extraction_prompt.yaml
│   └── labels.yaml
├── data/
│   ├── public_input/input/          # không commit dữ liệu nhạy cảm
│   ├── calibration/                 # nhãn thủ công nội bộ
│   ├── synthetic/                   # dữ liệu sinh và provenance
│   └── ontology/
│       ├── icd/                     # file BTC cung cấp hoặc file được phép dùng
│       └── rxnorm/                  # RRF release đã pin version
├── artifacts/
│   ├── ontology_index/
│   ├── faiss/
│   └── models/                      # weights local hoặc cache đóng gói
├── src/clinical_nlp/
│   ├── schema.py
│   ├── document_view.py
│   ├── sections.py
│   ├── rules.py
│   ├── llm_extractor.py
│   ├── aligner.py
│   ├── entity_resolver.py
│   ├── assertions.py
│   ├── ontology.py
│   ├── linkers.py
│   ├── pipeline.py
│   └── validator.py
├── scripts/
│   ├── doctor.py
│   ├── inspect_input.py
│   ├── build_ontology_index.py
│   ├── generate_weak_labels.py
│   ├── generate_synthetic.py
│   ├── train_ner.py
│   ├── train_assertion.py
│   ├── run_inference.py
│   ├── validate_output.py
│   └── package_submission.py
├── tests/
│   ├── test_offsets.py
│   ├── test_assertions.py
│   ├── test_schema.py
│   └── fixtures/
└── outputs/
```

Nguyên tắc: `src/` không biết tên file public; `configs/` không chứa absolute path của máy phát triển; mọi model/index cần có version và checksum trong manifest.

### C. Môi trường cài đặt

Khuyến nghị chạy Linux/WSL có CUDA gần với máy chấm. Với Windows native, FAISS/GPU và một số backend serving có thể khác; cần dùng Docker hoặc WSL nếu BTC chạy Linux.

```bash
git clone <repo-cua-doi>
cd medical-nlp-contest
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Cài PyTorch đúng CUDA của máy trước, sau đó:
pip install "transformers>=4.51.0" accelerate safetensors
pip install pydantic orjson regex rapidfuzz rank-bm25 tqdm pyyaml
pip install sentence-transformers faiss-cpu
pip install spacy medspacy
```

Nếu dùng GPU FAISS, thay `faiss-cpu` bằng package phù hợp với CUDA/OS. Nếu dùng vLLM:

```bash
pip install vllm
```

Không dùng `pip install -U` trong script inference. Sau khi chốt môi trường:

```bash
pip freeze > requirements.lock.txt
python scripts/doctor.py  # kiểm tra torch/cuda/model/index/schema
```

### D. Cấu hình tập trung

`configs/inference.yaml` nên có dạng:

```yaml
labels:
  entity_types:
    - TRIỆU_CHỨNG
    - TÊN_XÉT_NGHIỆM
    - KẾT_QUẢ_XÉT_NGHIỆM
    - CHẨN_ĐOÁN
    - THUỐC
  assertion_types:
    - TRIỆU_CHỨNG
    - CHẨN_ĐOÁN
    - THUỐC
  assertions: [isNegated, isFamily, isHistorical]

runtime:
  seed: 42
  max_document_chars: 20000
  llm_timeout_seconds: 45
  max_new_tokens: 1800
  llm_thinking: false
  verify_ambiguous_only: true

models:
  extractor: Qwen/Qwen3-8B
  extractor_revision: <pin-commit-or-snapshot>
  embedding: Qwen/Qwen3-Embedding-0.6B
  reranker: null
  quantization: <fp16-or-awq-or-gguf>

ontology:
  icd_path: data/ontology/icd/concepts.jsonl
  rxnorm_path: data/ontology/rxnorm/concepts.jsonl
  vector_index_dir: artifacts/faiss
  manifest: artifacts/ontology_index/manifest.json

candidate_policy:
  icd_max_k: 3
  rxnorm_max_k: 2
  min_margin_for_second: 0.04
  exact_alias_k: 1
```

Các giá trị `max_k`, `min_margin_for_second` chỉ là điểm bắt đầu. Phải thay bằng kết quả calibration nội bộ.

### E. Schema nội bộ và raw span

Không dùng output JSON của BTC làm object nội bộ duy nhất. Dùng object có thêm thông tin debug, sau đó serialize bỏ field nội bộ:

```python
# src/clinical_nlp/schema.py
from dataclasses import dataclass, field

ENTITY_TYPES = {
    "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN", "THUỐC",
}
ASSERTION_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
ALLOWED_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}

@dataclass
class Entity:
    text: str
    entity_type: str
    start: int
    end: int
    assertions: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    source: set[str] = field(default_factory=set)
    confidence: float = 0.0
    context: str = ""

    def to_public_json(self) -> dict:
        out = {
            "text": self.text,
            "type": self.entity_type,
            "position": [self.start, self.end],
        }
        if self.entity_type in ASSERTION_TYPES:
            out["assertions"] = self.assertions
        if self.entity_type in {"CHẨN_ĐOÁN", "THUỐC"}:
            out["candidates"] = self.candidates
        return out
```

`Entity.text` phải được tạo sau alignment:

```python
def make_entity(raw: str, start: int, end: int, typ: str) -> Entity:
    assert typ in ENTITY_TYPES
    assert 0 <= start < end <= len(raw)
    return Entity(text=raw[start:end], entity_type=typ,
                  start=start, end=end)
```

### F. Document view và offset map

```python
# src/clinical_nlp/document_view.py
from dataclasses import dataclass
import re

@dataclass
class Line:
    line_id: str
    raw_start: int
    raw_end: int
    raw_text: str
    search_text: str

def normalize_for_search(s: str) -> str:
    s = s.casefold()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def build_lines(raw: str) -> list[Line]:
    lines, cursor = [], 0
    for i, part in enumerate(raw.splitlines(keepends=True)):
        body = part.rstrip("\r\n")
        start, end = cursor, cursor + len(body)
        lines.append(Line(f"L{i:03d}", start, end, body,
                          normalize_for_search(body)))
        cursor += len(part)
    if not lines and raw == "":
        return []
    return lines

def exact_occurrences(raw: str, text: str, start_at: int = 0):
    pos = start_at
    while True:
        pos = raw.find(text, pos)
        if pos < 0:
            return
        yield pos, pos + len(text)
        pos += max(1, len(text))
```

Không thêm `[L000]` vào `raw`. Line marker chỉ nằm trong prompt. Nếu source input có newline Windows, giữ nguyên `\r\n` khi tính offset; không đọc bằng một hàm tự động đổi newline rồi mới tính vị trí.

### G. Section detector và assertion rules

Section detector nên trả cả `section_type`, `start`, `end`, `parent` và `confidence`. Không dùng một regex duy nhất cho toàn bộ hồ sơ.

```python
# src/clinical_nlp/sections.py
import re
from .document_view import normalize_for_search

HISTORY = [
    r"^\s*tiền sử bệnh\s*$",
    r"^\s*tiền sử phẫu thuật",
    r"thuốc trước khi nhập viện",
    r"các bệnh đã điều trị trước đây",
]
CURRENT = [
    r"bệnh sử hiện tại",
    r"tiền sử bệnh hiện tại",
    r"triệu chứng hiện tại",
    r"lý do nhập viện",
    r"tình trạng lúc vào viện",
]

def section_flags(heading: str) -> dict:
    h = normalize_for_search(heading)
    # CURRENT phải được xét trước HISTORY để tránh bắt "tiền sử ... hiện tại".
    if any(re.search(p, h) for p in CURRENT):
        return {"historical_default": False, "current": True}
    if any(re.search(p, h) for p in HISTORY):
        return {"historical_default": True, "current": False}
    return {"historical_default": False, "current": False}
```

Assertion resolver kết hợp entity-local cue, sentence cue và section scope:

```python
def resolve_assertions(entity, sentence, section):
    flags = set()
    if entity.entity_type not in ASSERTION_TYPES:
        return []

    if has_negation_scope(sentence, entity.start, entity.end):
        flags.add("isNegated")
    if has_family_relation_scope(sentence, entity.start, entity.end):
        flags.add("isFamily")
    if section.historical_default and not section.current:
        if has_history_scope(sentence, entity.start, entity.end):
            flags.add("isHistorical")

    # Cue hiện tại override historical lan từ section.
    if has_current_scope(sentence, entity.start, entity.end):
        flags.discard("isHistorical")
    return [x for x in ("isNegated", "isFamily", "isHistorical") if x in flags]
```

Khi dùng medspaCy, chỉ dùng nó như framework tokenizer/ConText; rule Vietnamese, scope boundary và section mapping vẫn phải tự viết. Đừng giả định rule tiếng Anh có thể dùng trực tiếp trên 100 file mẫu.

### H. LLM extractor local

#### H.1. Chạy Qwen3 trực tiếp bằng Transformers

```python
# src/clinical_nlp/llm_extractor.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalExtractor:
    def __init__(self, model_name, revision=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision,
            torch_dtype=torch.float16, device_map="auto"
        ).eval()

    def generate(self, prompt, max_new_tokens=1800):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        batch = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **batch, max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[0][batch.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

def parse_json_array(text: str) -> list[dict]:
    # Production code nên dùng JSON constrained decoding nếu backend hỗ trợ.
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    value = json.loads(text[start:end + 1])
    return value if isinstance(value, list) else []
```

Qwen3 chính thức hỗ trợ cả local inference và non-thinking/thinking mode. Với extraction JSON, chọn non-thinking, `do_sample=False`, giới hạn token và validate output; không để model gọi mạng hay tự chọn tool.

#### H.2. Prompt cố định

Prompt phải yêu cầu LLM trả **span nguyên bản**, không sửa chính tả, không sinh position:

```text
Bạn là module trích xuất thông tin y khoa.
Chỉ được copy text có thật trong INPUT. Không sửa dấu, không dịch,
không gộp hai occurrence ở hai vị trí.

TYPE hợp lệ: TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM,
KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC.
ASSERTION hợp lệ: isNegated, isFamily, isHistorical.

Không trả position. Hãy trả JSON array thuần túy:
[{"line_id":"L002","text":"...","type":"...",
  "assertions":[]}]

Không đưa heading vào entity. Với thuốc, giữ đúng span thuốc như trong
input. Với result, chỉ trích xuất khi có test/measurement context.
INPUT:
{numbered_raw_lines}
```

Pass 1 chỉ chạy trên document/chunk. Pass 2 chỉ chạy khi dictionary, LLM và resolver không đồng ý; không chạy verification toàn bộ 100 file nếu chưa benchmark thời gian.

#### H.3. Serving bằng vLLM tùy chọn

Nếu dùng server local:

```bash
vllm serve Qwen/Qwen3-8B \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

Inference client chỉ gọi `localhost`; không dùng endpoint ngoài. Cần pin vLLM version và test `enable_thinking=False` với đúng backend trước khi đóng gói. Nếu server không đảm bảo JSON constraint, vẫn phải chạy parser/validator ở client.

### I. Drug/test rule engine

#### I.1. Medication parser

```python
import re

DRUG_ATTR_RE = re.compile(
    r"(?ix)\b(?:\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|µg|ml|%)\b)"
    r"|\b(?:po|oral|iv|im|sc|sl|daily|bid|tid|qid|qam|qhs|prn|q\d+h)\b"
)

def medication_window(raw, drug_match):
    # Mở rộng quanh drug name nhưng dừng ở dấu câu/cue điều trị.
    # Không yêu cầu bắt buộc phải có numeric strength.
    left = find_item_start(raw, drug_match.start())
    right = find_item_end(raw, drug_match.end())
    return left, right
```

Thực tế nên dùng RxNorm aliases để tìm drug name trước, sau đó mở rộng span theo route/strength/frequency. Regex chỉ phát hiện thuộc tính, không tự biến một con số đứng riêng thành `THUỐC`.

#### I.2. Lab/result parser

```python
RESULT_PATTERNS = [
    re.compile(r"(?P<test>[A-Za-zÀ-ỹ%][^:;=]{0,80})\s*:\s*(?P<value>[^;,.\n]+)"),
    re.compile(r"(?P<test>[A-Za-zÀ-ỹ%][^:;=]{0,80})\s+(?:là|=)\s+(?P<value>[^;,.\n]+)"),
    re.compile(r"(?P<test>[^,;\n]{2,80})\s+(?P<value>tăng nhẹ|tăng|giảm|âm tính|dương tính)")
]

def parse_lab_results(raw, known_tests):
    entities = []
    for pattern in RESULT_PATTERNS:
        for m in pattern.finditer(raw):
            test = m.group("test").strip()
            value = m.group("value").strip()
            if is_known_test(test, known_tests) or has_lab_context(raw, m.start()):
                entities.append((test, value, m.start("test"), m.end("test")))
                entities.append((value, "KẾT_QUẢ_XÉT_NGHIỆM",
                                 m.start("value"), m.end("value")))
    return entities
```

Đây là skeleton; boundary thực tế phải sửa theo calibration. Không dùng `\b\d+` đơn độc vì bộ input có rất nhiều số tuổi, thời gian, liều và kích thước.

### J. Entity merge và overlap resolver

Mỗi detector trả `source`, `confidence`, `span`, `type`. Resolver:

```python
def merge_candidates(items):
    groups = group_by_exact_span(items)
    merged = []
    for group in groups:
        typ = choose_type(group)  # ưu tiên bằng chứng mạnh + context
        e = best_item(group, typ)
        e.source = {s for item in group for s in item.source}
        e.confidence = combine_confidence(group)
        merged.append(e)
    return resolve_overlaps(merged)
```

Quy tắc overlap cần explicit:

1. cùng span, khác detector → merge;
2. cùng text nhưng vị trí khác → giữ cả hai;
3. result value nằm trong drug dose → ưu tiên drug context, không sinh lab result;
4. heading nằm trong span semantic → cắt heading;
5. span dài và span ngắn cùng type → chọn theo guideline boundary đã calibration, không luôn chọn dài nhất.

### K. Ontology builder và retrieval

#### K.1. Chuẩn hóa format concept nội bộ

ICD:

```json
{"code":"K21.9","label":"Gastro-esophageal reflux disease without esophagitis",
 "aliases":["trào ngược dạ dày thực quản"],"parents":["K21"]}
```

RxNorm:

```json
{"rxcui":"360047","tty":"SCD","name":"chlorpheniramine 0.4 MG/ML",
 "ingredient":"chlorpheniramine","strength":"0.4 MG/ML",
 "dose_form":"solution"}
```

Không tự điền `name_vi` bằng dịch máy vào field gold. Nếu cần tiếng Việt để retrieval, lưu riêng `retrieval_aliases` và giữ label nguồn.

#### K.2. Build lexical index

```python
def concept_text(c):
    return " | ".join([c["label"], *c.get("aliases", []), c.get("name", "")])

concepts = load_concepts_from_pinned_release()
bm25 = BM25Okapi([tokenize(concept_text(c)) for c in concepts])
save_pickle(bm25, "artifacts/ontology_index/bm25.pkl")
write_jsonl(concepts, "data/ontology/rxnorm/concepts.jsonl")
```

#### K.3. Build dense index tùy chọn

Qwen3-Embedding-0.6B cần dùng instruction phù hợp cho query và encode concept documents nhất quán. Precompute embeddings một lần:

```python
query_instruction = (
    "Retrieve the ICD-10 or RxNorm concept that best normalizes the medical mention."
)
query = f"Instruct: {query_instruction}\nQuery: {mention}"
query_vec = encoder.encode(query, normalize_embeddings=True)
scores, ids = faiss_index.search(query_vec[None, :], 30)
```

FAISS chỉ làm candidate retrieval. Không coi cosine score là xác suất đúng; reranker/policy vẫn phải kiểm tra ingredient, strength, parent code và context.

#### K.4. Linker API

```python
class ICDLinker:
    def link(self, mention, context, max_k=3):
        pool = union(
            self.exact_alias.search(mention),
            self.bm25.search(mention, k=30),
            self.faiss.search(mention, k=30),
        )
        ranked = self.rerank(mention, context, pool)
        return calibrated_select(ranked, max_k=max_k, kind="icd")

class RxNormLinker:
    def link(self, mention, context, max_k=2):
        attrs = parse_medication_attributes(mention)
        pool = self.lookup_exact(attrs)
        pool += self.bm25.search(mention, k=30)
        pool += self.faiss.search(mention, k=30)
        ranked = self.score_drug_attributes(attrs, context, unique(pool))
        return calibrated_select(ranked, max_k=max_k, kind="rxnorm")
```

### L. Weak labels, synthetic data và fine-tuning

Vì đề yêu cầu tạo thêm dữ liệu, nên submission nên có nhánh data-generation tái lập được, dù inference cuối có thể dùng student model hoặc hybrid.

#### L.1. Weak-label record

```json
{
  "text":"metoprolol 25mg po bid",
  "start":12,"end":38,"type":"THUỐC",
  "assertions":["isHistorical"],
  "candidates":["866436"],
  "provenance":["rxnorm_exact","section_history"],
  "confidence":0.96
}
```

Chỉ train với record có provenance và confidence đủ cao. Giữ một bộ `manual_dev` không được dùng để sinh rule mới sau mỗi vòng tuning.

#### L.2. BIOES NER student

```text
Qwen3/rules teacher
        ↓
weak/synthetic BIOES labels
        ↓
ViPubmedDeBERTa hoặc XLM-R + token classification head
        ↓
teacher-student ensemble trên span
```

Không gọi checkpoint ViPubmedDeBERTa/XLM-R gốc là NER. Phải tạo `AutoModelForTokenClassification` với đúng số label và train trên label của cuộc thi.

Ví dụ label mapping:

```python
LABELS = ["O"]
for typ in ENTITY_TYPES:
    LABELS += [f"B-{typ}", f"I-{typ}", f"E-{typ}", f"S-{typ}"]
```

Sau fine-tune, dùng student để tăng recall nhanh; vẫn chạy exact alignment và overlap resolver. Chỉ giữ ensemble nếu `text/type` trên manual-dev tăng mà assertion/candidate không giảm.

#### L.3. Assertion student

Assertion nên train theo entity-context pair, không phải toàn văn:

```json
{"entity":"khó thở",
 "left_context":"không ghi nhận triệu chứng",
 "right_context":"hoặc khàn tiếng",
 "section":"Bệnh sử hiện tại",
 "labels":["isNegated"]}
```

XLM-R hoặc encoder nhỏ hơn có thể phân loại multi-label. Rule vẫn là fallback khi confidence thấp.

### M. Scripts và lệnh chạy đầy đủ

```bash
# 1. Kiểm tra input và tạo manifest
python scripts/inspect_input.py --input_dir test/input \
  --manifest artifacts/input_manifest.json

# 2. Build index một lần, không chạy trong mỗi document
python scripts/build_ontology_index.py \
  --icd data/ontology/icd/source.json \
  --rxnorm data/ontology/rxnorm/RXNCONSO.RRF \
  --output artifacts/ontology_index \
  --embedding_model Qwen/Qwen3-Embedding-0.6B

# 3. Sinh weak/synthetic data để huấn luyện/calibration
python scripts/generate_weak_labels.py \
  --input_dir test/input --ontology artifacts/ontology_index \
  --output_dir data/weak
python scripts/generate_synthetic.py \
  --ontology artifacts/ontology_index --n 10000 \
  --output_dir data/synthetic --seed 42

# 4. Tùy chọn train student
python scripts/train_ner.py --config configs/train_ner.yaml
python scripts/train_assertion.py --config configs/train_assertion.yaml

# 5. Inference một lệnh
python scripts/run_inference.py \
  --input_dir test/input \
  --output_dir output \
  --config configs/inference.yaml \
  --model_dir artifacts/models

# 6. Validate trước khi nộp
python scripts/validate_output.py \
  --input_dir test/input --output_dir output --strict

# 7. Đóng gói
python scripts/package_submission.py \
  --output_dir output --zip_path output.zip
```

`run_inference.py` phải tự tạo `1.json` đến `100.json`; không phụ thuộc việc người dùng chạy từng file. `package_submission.py` phải kiểm tra archive path trước khi zip để tránh tạo `output/output/` hoặc thiếu thư mục `output/`.

### N. Validator đầy đủ

```python
def validate_entity(e, raw):
    assert set(e) >= {"text", "type", "position"}
    assert e["type"] in ENTITY_TYPES
    start, end = e["position"]
    assert isinstance(start, int) and isinstance(end, int)
    assert 0 <= start < end <= len(raw)
    assert raw[start:end] == e["text"]

    if e["type"] in ASSERTION_TYPES:
        assert set(e.get("assertions", [])) <= ALLOWED_ASSERTIONS
    else:
        assert "assertions" not in e

    if e["type"] in {"CHẨN_ĐOÁN", "THUỐC"}:
        assert all(isinstance(c, str) for c in e.get("candidates", []))
    else:
        assert "candidates" not in e
```

Trong giai đoạn debug, nên lưu thêm `debug.jsonl` chứa source/confidence/rule hits; không đưa debug field vào file nộp.

### O. Test tối thiểu phải có

```python
def test_repeated_occurrence():
    raw = "khó thở nhưng không khó thở ở người nhà"
    # Hai occurrence có span khác nhau; assertion khác nhau.

def test_current_overrides_history():
    # "Tiền sử bệnh hiện tại" không được mặc định historical.

def test_raw_offset_with_newline():
    # Kiểm tra CRLF/LF và line marker không làm lệch offset.

def test_drug_without_strength():
    # Guaifenesin ml po q6h:prn vẫn phải được xem là candidate thuốc.

def test_textual_lab_result():
    # "CEA tăng nhẹ lên 4.9", "âm tính" không bị giới hạn bởi numeric regex.

def test_schema_fields_by_type():
    # Lab/test không có field không áp dụng; drug/diagnosis có candidates.
```

Chạy:

```bash
pytest -q
python scripts/run_inference.py --input_dir test/input --output_dir output --config configs/inference.yaml
python scripts/validate_output.py --input_dir test/input --output_dir output --strict
```

### P. Đo runtime và ablation

Ghi riêng thời gian:

```text
load_model
build_document_view
rules_and_dictionary
llm_extraction
alignment_and_assertion
icd_linking
rxnorm_linking
serialization
```

Chạy ít nhất các ablation sau trên manual-dev:

```text
rules only
rules + LLM
rules + LLM + dense retrieval
rules + LLM + reranker
rules + student + LLM fallback
```

Chọn cấu hình theo điểm composite và runtime, không theo cảm giác model lớn hơn luôn tốt hơn. Với giới hạn 600 giây, cache ontology và tránh gọi LLM cho entity đã được rule xác định chắc chắn thường có tác động lớn hơn việc thêm một reranker.

## 11. Những nội dung của bản cũ cần giữ, sửa hoặc bỏ

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

## 12. Checklist trước khi nộp

- [ ] Có 100 file JSON đúng tên và đúng thư mục.
- [ ] `text` luôn là substring nguyên bản.
- [ ] Position đã xác minh là `[start, end)` theo output mẫu.
- [ ] Không gán historical cho toàn bộ `Bệnh sử hiện tại`.
- [ ] Có negation scope cho list và câu phủ định.
- [ ] Có family scope, không chỉ keyword matching.
- [ ] Bắt được thuốc không có strength.
- [ ] Bắt được result dạng số, chữ, âm tính/dương tính và phrase có đơn vị.
- [ ] Không gán mọi imaging/procedure thành lab test nếu guideline không yêu cầu.
- [ ] Candidate count được calibration; không top-5 cứng.
- [ ] Ontology release và checksum cố định.
- [ ] Không gọi API ngoài trong runtime.
- [ ] Có model weights, code, requirements lock và README.
- [ ] Runtime 100 file dưới 600 giây trên môi trường gần máy chấm.
- [ ] Chạy private-like test với văn bản mới, không hardcode tên file public.

## Kết luận cuối

Pipeline phù hợp nhất cho điều kiện này là:

```text
rules/dictionary có độ chính xác cao
local LLM zero/few-shot cho semantic ambiguity
exact raw-span alignment
assertion resolver có scope
ICD/RxNorm retrieval offline theo version cố định
calibrated candidate selection
weak/synthetic data để huấn luyện và calibration
strict schema/runtime validator
```

Đây là bản tương thích hơn với 100 input thực tế, private test, metric WER/Jaccard, giới hạn model local và yêu cầu BTC dựng lại source code. Không nên triển khai bản cũ nguyên trạng vì bốn lỗi `historical`, lab result, candidate count và ngân sách model có thể làm giảm điểm trực tiếp.

### Nguồn kỹ thuật cần pin trong README

- [Qwen3 official repository](https://github.com/QwenLM/Qwen3)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [RxNorm official files and release documentation](https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html)
- [RxNorm technical documentation](https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html)
- [WHO ICD-10 browser/documentation](https://icd.who.int/browse10/)
