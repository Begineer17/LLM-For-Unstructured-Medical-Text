import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_nlp_pipeline import (  # noqa: E402
    Entity,
    Ontology,
    align_llm,
    detect,
    find_all,
    link_entities,
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

    def test_generic_numbered_heading_resets_history_scope(self):
        raw = "Tiền sử bệnh\n- metoprolol\n3. Đánh giá tại bệnh viện\nKhám lâm sàng:\n- đánh trống ngực"
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

    def test_merged_test_result_is_split_without_name_entity(self):
        raw = "chụp x-quang ngực không ghi nhận gì bất thường; phân tích nước tiểu không có gì đáng chú ý"
        first_end = raw.index(";")
        second_start = first_end + 2
        result = resolve(raw, [
            Entity(raw[:first_end], "KẾT_QUẢ_XÉT_NGHIỆM", 0, first_end, confidence=.55, source="llm"),
            Entity(raw[second_start:], "KẾT_QUẢ_XÉT_NGHIỆM", second_start, len(raw), confidence=.55, source="llm"),
        ])
        self.assertEqual([(item.text, item.typ) for item in result], [
            ("chụp x-quang ngực", "TÊN_XÉT_NGHIỆM"),
            ("không ghi nhận gì bất thường", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ("phân tích nước tiểu", "TÊN_XÉT_NGHIỆM"),
            ("không có gì đáng chú ý", "KẾT_QUẢ_XÉT_NGHIỆM"),
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

    def test_final_linker_links_entity_after_resolve(self):
        raw = "đã dùng coumadin"
        start = raw.index("đã dùng coumadin")
        resolved = resolve(raw, [Entity(raw[start:], "THUỐC", start, len(raw), candidates=["bad"])])
        icd = Ontology("data/ontology/icd/concepts.jsonl", kind="icd")
        rx = Ontology("data/ontology/rxnorm/concepts.jsonl", kind="rxnorm")
        link_entities(resolved, icd, rx)
        self.assertEqual([(item.text, item.candidates) for item in resolved], [("coumadin", ["11289"])])

    def test_internal_link_hint_preserves_raw_span_and_uses_offline_icd_lookup(self):
        raw = "Nhiễm virus Herpes simplex"
        entity = Entity(
            raw,
            "CHẨN_ĐOÁN",
            0,
            len(raw),
            source="llm",
            link_text="Herpesviral infection, unspecified",
        )
        link_entities(
            [entity],
            Ontology("data/ontology/icd/concepts.jsonl", kind="icd"),
            Ontology("data/ontology/rxnorm/concepts.jsonl", kind="rxnorm"),
        )
        self.assertEqual(entity.text, raw)
        self.assertEqual(entity.candidates, ["B00.9"])

    def test_align_llm_keeps_internal_link_hint_out_of_public_schema(self):
        raw = "Nhiễm virus Herpes simplex"
        entities = align_llm(raw, [{
            "line_id": "L001",
            "text": raw,
            "type": "CHẨN_ĐOÁN",
            "link_text": "Herpesviral infection, unspecified",
        }])
        self.assertEqual(entities[0].link_text, "Herpesviral infection, unspecified")
        self.assertNotIn("link_text", entities[0].public())

    def test_rxnorm_brand_and_combination_aliases(self):
        rx = Ontology("data/ontology/rxnorm/concepts.jsonl", kind="rxnorm")
        self.assertEqual(rx.lookup("Coumadin"), ["11289"])
        self.assertEqual(rx.lookup("albuterolipratropium nebs x2"), ["214199"])
        self.assertEqual(rx.lookup("aspirin 325mg"), ["1191"])
        self.assertEqual(rx.lookup("not a medicine"), [])

    def test_rxnorm_semantic_base_ingredient_and_sig_variants(self):
        rx = Ontology("data/ontology/rxnorm/concepts.jsonl", kind="rxnorm")
        self.assertEqual(rx.lookup("senna 8.6 mg po bid:prn"), ["312935"])
        self.assertEqual(rx.lookup("pravastatin 40 mg po daily"), ["904475"])
        self.assertEqual(rx.lookup("metoprolol 25mg po bid"), ["1370489"])

    def test_icd_vietnamese_curated_aliases(self):
        icd = Ontology("data/ontology/icd/concepts.jsonl", kind="icd")
        self.assertEqual(icd.lookup("rung nhĩ kịch phát"), ["I48.0"])
        self.assertEqual(icd.lookup("béo phì"), ["E66.9"])
        self.assertEqual(icd.lookup("viêm túi mật cấp"), ["K81.0"])

    def test_icd_semantic_base_diagnosis_and_context_wrappers(self):
        icd = Ontology("data/ontology/icd/concepts.jsonl", kind="icd")
        self.assertEqual(icd.lookup("rung nhĩ"), ["I48.0"])
        self.assertEqual(icd.lookup("bệnh nhân bị béo phì"), ["E66.9"])
        self.assertEqual(icd.lookup("u cơ trơn tử cung, không đặc hiệu"), ["D25.9"])

    def test_measurement_result_is_split_from_imaging_finding(self):
        raw = (
            "Tử cung đo 14,2 x 8,8 x 7,5 cm; "
            "U cơ trơn lớn nhất ở đoạn dưới của tử cung đo 4,1 x 4,8 x 4,1 cm"
        )
        second_start = raw.index("U cơ trơn")
        items = resolve(raw, [
            Entity(raw[:raw.index(";")], "KẾT_QUẢ_XÉT_NGHIỆM", 0, raw.index(";")),
            Entity(raw[second_start:], "KẾT_QUẢ_XÉT_NGHIỆM", second_start, len(raw)),
        ])
        values = [(item.text, item.typ) for item in items]
        self.assertIn(("Tử cung đo", "TÊN_XÉT_NGHIỆM"), values)
        self.assertIn(("14,2 x 8,8 x 7,5 cm", "KẾT_QUẢ_XÉT_NGHIỆM"), values)
        self.assertIn(
            ("U cơ trơn lớn nhất ở đoạn dưới của tử cung đo", "TÊN_XÉT_NGHIỆM"),
            values,
        )
        self.assertIn(("4,1 x 4,8 x 4,1 cm", "KẾT_QUẢ_XÉT_NGHIỆM"), values)

    def test_medication_context_is_trimmed_and_non_drug_support_is_removed(self):
        raw = (
            "Tăng liều bactrim (do bác sĩ kê đơn)\n"
            "doxycycline (prescribed by primary care)\n"
            "Thở oxy tại nhà"
        )
        result = resolve(raw, [
            Entity("Tăng liều bactrim (do bác sĩ kê đơn)", "THUỐC", 0, raw.index("\n")),
            Entity(
                "doxycycline (prescribed by primary care)",
                "THUỐC",
                raw.index("doxycycline"),
                raw.rindex("\n"),
            ),
            Entity("Thở oxy tại nhà", "THUỐC", raw.rindex("Thở"), len(raw)),
        ])
        self.assertEqual([(item.text, item.typ) for item in result], [
            ("bactrim", "THUỐC"),
            ("doxycycline", "THUỐC"),
        ])

    def test_administrative_symptoms_are_removed_and_labelled_lists_are_atomic(self):
        raw = (
            "Khám tại phòng khám vào ngày nhập viện\n"
            "Được giới thiệu đến khoa cấp cứu để đánh giá thêm\n"
            "Các triệu chứng liên quan: ban đỏ, chảy mủ"
        )
        result = resolve(raw, [
            Entity("Khám tại phòng khám vào ngày nhập viện", "TRIỆU_CHỨNG", 0, raw.index("\n")),
            Entity(
                "Được giới thiệu đến khoa cấp cứu để đánh giá thêm",
                "TRIỆU_CHỨNG",
                raw.index("Được giới thiệu"),
                raw.rindex("\n"),
            ),
            Entity(
                "Các triệu chứng liên quan: ban đỏ, chảy mủ",
                "TRIỆU_CHỨNG",
                raw.rindex("Các triệu chứng"),
                len(raw),
            ),
        ])
        self.assertEqual([(item.text, item.typ) for item in result], [
            ("ban đỏ", "TRIỆU_CHỨNG"),
            ("chảy mủ", "TRIỆU_CHỨNG"),
        ])

    def test_procedure_and_heading_wrappers_are_removed(self):
        raw = "3. Đánh giá tại bệnh viện\ndùng vancomycin\ncắt bỏ tuyến vú trái"
        start = raw.index("3.")
        result = resolve(raw, [
            Entity("3. Đánh giá tại bệnh viện", "CHẨN_ĐOÁN", start, start + len("3. Đánh giá tại bệnh viện")),
            Entity("dùng vancomycin", "THUỐC", raw.index("dùng"), raw.index("dùng") + len("dùng vancomycin")),
            Entity("cắt bỏ tuyến vú trái", "THUỐC", raw.index("cắt"), raw.index("cắt") + len("cắt bỏ tuyến vú trái")),
        ])
        self.assertEqual([(item.text, item.typ) for item in result], [("vancomycin", "THUỐC")])

    def test_linker_clears_candidates_for_non_linkable_types(self):
        entities = [Entity("aspirin", "TRIỆU_CHỨNG", 0, 7, candidates=["1191"])]
        link_entities(entities, Ontology(None, "icd"), Ontology(None, "rxnorm"))
        self.assertEqual(entities[0].candidates, [])


if __name__ == "__main__":
    unittest.main()
