import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_nlp_pipeline import (  # noqa: E402
    Entity,
    Ontology,
    detect,
    find_all,
    resolve,
    validate_pair,
)


class PipelineRegressionTests(unittest.TestCase):
    def test_same_span_different_test_types_is_adjudicated(self):
        raw = "monitor holter"
        start = raw.index("monitor holter")
        result = resolve(raw, [
            Entity(raw[start:start + 14], "KẾT_QUẢ_XÉT_NGHIỆM", start, start + 14, confidence=.55, source="llm"),
            Entity(raw[start:start + 14], "TÊN_XÉT_NGHIỆM", start, start + 14, confidence=.55, source="llm"),
        ])
        self.assertEqual([(item.text, item.typ) for item in result], [(raw, "TÊN_XÉT_NGHIỆM")])

    def test_repeated_occurrences_are_kept_and_scoped(self):
        raw = "đánh trống ngực nhưng không đánh trống ngực"
        first = raw.index("đánh trống ngực")
        second = raw.index("đánh trống ngực", first + 1)
        result = resolve(raw, [
            Entity(raw[first:first + 15], "TRIỆU_CHỨNG", first, first + 15),
            Entity(raw[second:second + 15], "TRIỆU_CHỨNG", second, second + 15),
        ])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].assertions, [])
        self.assertEqual(result[1].assertions, ["isNegated"])

    def test_current_history_heading_does_not_leak_historical(self):
        raw = "1. Tiền sử bệnh\n- metoprolol\n2. Tiền sử bệnh hiện tại\n- đánh trống ngực"
        start = raw.rindex("đánh trống ngực")
        result = resolve(raw, [Entity(raw[start:start + 15], "TRIỆU_CHỨNG", start, start + 15)])
        self.assertEqual(result[0].assertions, [])

    def test_treatment_failure_does_not_negate_medication(self):
        raw = "Bắt đầu dùng metoprolol 25mg po bid, không có cải thiện"
        start = raw.index("metoprolol")
        end = raw.index(",", start)
        result = resolve(raw, [Entity(raw[start:end], "THUỐC", start, end)])
        self.assertNotIn("isNegated", result[0].assertions)

    def test_negation_scope_covers_coordinated_list(self):
        raw = "Không buồn nôn, hay nôn, đổ mồ hôi"
        spans = [
            (raw.index("buồn nôn"), len("buồn nôn")),
            (raw.index("nôn", raw.index("buồn nôn") + len("buồn nôn")), len("nôn")),
            (raw.index("đổ mồ hôi"), len("đổ mồ hôi")),
        ]
        result = resolve(raw, [Entity(raw[start:start + length], "TRIỆU_CHỨNG", start, start + length) for start, length in spans])
        self.assertEqual(len(result), 3)
        self.assertTrue(all("isNegated" in item.assertions for item in result))

    def test_heading_is_not_an_entity(self):
        raw = "Kết quả khám lâm sàng\nđánh trống ngực"
        heading_start = raw.index("khám lâm sàng")
        symptom_start = raw.index("đánh trống ngực")
        result = resolve(raw, [
            Entity("khám lâm sàng", "CHẨN_ĐOÁN", heading_start, heading_start + len("khám lâm sàng")),
            Entity(raw[symptom_start:symptom_start + 15], "TRIỆU_CHỨNG", symptom_start, symptom_start + 15),
        ])
        self.assertEqual([item.text for item in result], ["đánh trống ngực"])

    def test_test_name_and_result_are_separate(self):
        raw = "chụp x-quang ngực không ghi nhận gì bất thường"
        name_end = raw.index(" không")
        result = resolve(raw, [
            Entity(raw[:name_end], "TÊN_XÉT_NGHIỆM", 0, name_end),
            Entity(raw, "KẾT_QUẢ_XÉT_NGHIỆM", 0, len(raw), confidence=.55, source="llm"),
        ])
        self.assertEqual([(item.text, item.typ) for item in result], [
            ("chụp x-quang ngực", "TÊN_XÉT_NGHIỆM"),
            ("không ghi nhận gì bất thường", "KẾT_QUẢ_XÉT_NGHIỆM"),
        ])

    def test_crlf_offsets_and_schema(self):
        raw = "WBC: 14,43\r\n"
        data = [{"text": "WBC", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 3]}]
        self.assertEqual(validate_pair(raw, data), [])

    def test_drug_without_strength_is_detected(self):
        rx = Ontology(str(Path(__file__).parent / "fixtures" / "rxnorm.jsonl"), kind="rxnorm")
        result = detect("metoprolol po bid", Ontology(None), rx)
        self.assertTrue(any(item.typ == "THUỐC" for item in result))

    def test_numeric_and_textual_lab_results(self):
        raw = "WBC: 14,43; CEA bình thường"
        result = detect(raw, Ontology(None), Ontology(None), {"WBC", "CEA"})
        values = {(item.text, item.typ) for item in result}
        self.assertIn(("14,43", "KẾT_QUẢ_XÉT_NGHIỆM"), values)
        self.assertIn(("bình thường", "KẾT_QUẢ_XÉT_NGHIỆM"), values)


if __name__ == "__main__":
    unittest.main()
