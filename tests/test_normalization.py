import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.attrition_diagnostics import _normalization_audit
from data.normalize_data import main, normalize_date


class TestDateNormalization(unittest.TestCase):
    def test_precise_years_are_retained_without_dictionary_entries(self):
        for value in (1172, "1499", "1650", " 1500 "):
            with self.subTest(value=value):
                self.assertEqual(normalize_date(value, {}), int(str(value).strip()))

    def test_precise_year_takes_precedence_over_dictionary(self):
        self.assertEqual(normalize_date("1500", {"1500": 1450}), 1500)

    def test_other_dating_statements_keep_dictionary_semantics(self):
        mapping = {"15,1": 1425, "1499/1501": 1500, "unknown": None}
        self.assertEqual(normalize_date("15,1", mapping), 1425)
        self.assertEqual(normalize_date("1499/1501", mapping), 1500)
        for value in ("unknown", "um 1500", "1500.0", "", None, float("nan"), pd.NA):
            with self.subTest(value=value):
                self.assertIsNone(normalize_date(value, mapping))

    def test_normalization_and_audit_retain_years_and_preserve_other_gates(self):
        def row(token, **changes):
            return {
                "lemma_id": 1,
                "date": "1500",
                "language-region": "test",
                "infl": "3.Sg.Prät.Ind",
                "document_id": "ENHG:D1",
                "token_id": token,
                "observation_id": f"ENHG:D1:{token}",
                "specific_dating": "",
                "time": "",
                "grapho": "",
                **changes,
            }

        rows = [row(f"t{i}") for i in range(12)]
        rows += [
            row("mapped-date", date="15,1"),
            row("unknown-date", date="unknown"),
            row("unknown-variety", **{"language-region": "unknown"}),
            row("unknown-slot", infl="--"),
            row("unmapped-family", lemma_id=None),
            row("rare-family", lemma_id=2),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(rows).to_csv(root / "input.csv", index=False)
            (root / "dates.json").write_text(json.dumps([
                {"original": "15,1", "normalized": 1425},
            ]))
            (root / "dialects.json").write_text(json.dumps([
                {"original": "test", "normalized": "Upper German"},
            ]))
            main(argparse.Namespace(
                input=root / "input.csv",
                date_mapping=root / "dates.json",
                dialect_mapping=root / "dialects.json",
                output=root / "normalized.csv",
            ))
            normalized = pd.read_csv(root / "normalized.csv")
            combined = pd.read_csv(root / "input.csv", dtype=str)
            *_, audited = _normalization_audit(
                combined, root / "dialects.json", root / "dates.json"
            )

        expected_ids = {f"ENHG:D1:t{i}" for i in range(12)} | {"ENHG:D1:mapped-date"}
        self.assertEqual(set(normalized["observation_id"]), expected_ids)
        self.assertEqual(set(audited["observation_id"]), expected_ids)
        dates = normalized.set_index("token_id")["date"]
        self.assertEqual(dates["t0"], 1500)
        self.assertEqual(dates["mapped-date"], 1425)
        self.assertEqual(set(normalized["lemma_id"]), {1})


if __name__ == "__main__":
    unittest.main()
