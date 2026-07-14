# Clinical NLP contest pipeline

Pipeline cho bài toán chuẩn hóa văn bản lâm sàng tiếng Việt/Anh. Mỗi file `input/{n}.txt` được biến thành đúng một `output/{n}.json` (JSON array), giữ nguyên substring và offset `[start, end)` của văn bản gốc. Không có nhãn bệnh/thuốc/xét nghiệm hard-code trong source: ontology và LLM là đầu vào triển khai bắt buộc.

## Điểm chính

- Nhận diện thuốc bằng alias/index RxNorm phiên bản cố định (tên hoạt chất/brand, liều, dạng bào chế, route/frequency, kể cả thuốc không có strength).
- Nhận diện triệu chứng, xét nghiệm, phủ định, tiền sử và quan hệ gia đình bằng LLM local; có thể bổ sung dictionary riêng qua CLI nhưng không nằm trong code.
- Assertion theo cửa sổ câu + section: `isNegated`, `isFamily`, `isHistorical`; không lan sang câu kế tiếp.
- Candidate linking offline: exact alias trước, fuzzy fallback sau. Có thể nạp JSONL/CSV ICD và RxNorm phiên bản cố định.
- Tùy chọn semantic extraction bằng endpoint OpenAI-compatible chạy **localhost** (vLLM/Ollama/llama.cpp). LLM chỉ trả span text; client align lại về raw text và loại bỏ span không tồn tại.
- Validator kiểm tra schema, type, offset và text/position; CLI có thể đóng gói `output.zip`.

## Cài đặt

Python 3.10+ là đủ cho bộ điều phối; inference contest cần ontology files và local LLM:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

`rapidfuzz` là tùy chọn nhưng nên cài để fuzzy linking. Downloader dùng HTTP chuẩn Python; model local chỉ cần cài backend tương ứng trong môi trường riêng.

## Tải ontology chính thức

Chọn đúng biến thể mà BTC quy định (ICD-10 WHO, ICD-10-CM hoặc danh mục nội bộ). Script dưới đây dùng ICD-10-CM Code Descriptions FY2026 của CDC và RxNorm Prescribable Monthly Release của NLM, tạo JSONL cùng manifest SHA-256:

```bash
python scripts/download_ontologies.py --out data/ontology
```

RxNorm full release có thể yêu cầu UMLS license. Khi đã được cấp quyền, truyền URL `RxNorm_full_YYYYMMDD.zip` qua `--rxnorm-url`. Không gọi RxNav/API trong lúc inference; chỉ dùng index đã tải và pin release.

## Chạy trên bộ 100 mẫu

```bash
python scripts/run_inference.py \
  --icd data/ontology/icd/concepts.jsonl \
  --rxnorm data/ontology/rxnorm/concepts.jsonl \
  --llm-endpoint http://127.0.0.1:8000 \
  --llm-model Qwen/Qwen3-8B \
  --input input --output output --validate --zip
python scripts/validate_output.py --input input --output output
```

Lệnh tạo `output/1.json ... output/100.json` và `output.zip` với cấu trúc `output/{n}.json`. Các file rỗng vẫn được sinh thành `[]`.

### Nạp ontology đã tải

JSONL mỗi dòng có `code`, `label`, tùy chọn `aliases`; hoặc CSV có cột `code`/`rxcui` và `label`/`STR`:

```bash
python scripts/run_inference.py \
  --icd data/ontology/icd/concepts.jsonl \
  --rxnorm data/ontology/rxnorm/concepts.jsonl \
  --llm-endpoint http://127.0.0.1:8000 \
  --input input --output output --validate --zip
```

Không gọi API ICD/RxNorm trong lúc chấm. Hãy pin release, lưu checksum và chỉ đưa file ontology được BTC cho phép. `scripts/build_ontology_index.py` chuyển CSV/JSONL thành JSONL chuẩn và in SHA-256 manifest. Nếu không truyền ontology hoặc LLM, CLI sẽ dừng với lỗi thay vì chạy mock.

### Dùng LLM local

Khởi động server local (ví dụ vLLM) rồi truyền URL base:

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.9
python scripts/run_inference.py --llm-endpoint http://127.0.0.1:8000 \
  --llm-model Qwen/Qwen3-8B --input input --output output --validate
```

Client gọi duy nhất `http://127.0.0.1`; endpoint phải trả OpenAI-compatible JSON. Nếu server lỗi, pipeline ghi lỗi và không âm thầm chuyển sang mock. Dùng non-thinking, greedy decoding và giới hạn token để đáp ứng giới hạn thời gian. Qwen3-8B (hoặc model nhỏ hơn) phải được quantize/cấu hình theo quy định BTC; không ghép thêm model khiến tổng tham số vượt giới hạn.

## Sinh dữ liệu bổ sung

Đề yêu cầu tạo dữ liệu ngoài lời giải chính. Script mẫu sinh record có provenance để dùng cho huấn luyện/weak-labeling, không tự động trộn vào output:

```bash
python scripts/generate_synthetic.py --output data/synthetic --count 1000 --seed 42
```

Trước khi train, cần người kiểm tra hoặc verifier loại các mẫu sai. Có thể mở rộng template bằng alias/section/negation/family/history và lưu nhãn span bằng code, không lấy offset do LLM tự sinh.

## Schema và kiểm thử

Các type hợp lệ: `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`. Assertion chỉ xuất hiện với ba type đầu tương ứng; candidate chỉ xuất hiện với `CHẨN_ĐOÁN` và `THUỐC`. `position` luôn là Python slice trên raw UTF-8 string:

```python
assert entity["text"] == raw[entity["position"][0]:entity["position"][1]]
```

Nếu BTC cung cấp validator chính thức, chạy nó sau validator nội bộ. Nên lập calibration set 20–40 hồ sơ, đánh giá WER/Jaccard và điều chỉnh candidate-count/biên span theo dữ liệu private.

## Cấu trúc

`clinical_nlp_pipeline.py` chứa schema, ontology loader, detector, assertion resolver, local-LLM adapter, resolver và serializer. `scripts/` cung cấp CLI inference, validator, ontology downloader/builder và synthetic generator. Thiết kế không hard-code tên file, bệnh, thuốc hay sample cụ thể; mọi đường dẫn truyền qua CLI.
