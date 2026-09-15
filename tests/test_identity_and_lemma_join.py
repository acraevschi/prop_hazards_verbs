import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.consonant_analysis import build_paired_cells
from analysis.marking_type_summary import reshape
from data.extract_enhg_data import extract_strong_verbs as extract_enhg
from data.extract_mhg_data import extract_strong_verbs as extract_mhg
from data.lemmas.enhg_mhg_mapping import (
    combine_corpus_data,
    validate_lemma_mapping,
    write_output,
)


class TestCorpusIdentityExtraction(unittest.TestCase):
    def test_mhg_keeps_document_and_token_identity_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "mhg"
            corpus.mkdir()
            for document in ("M001", "M002"):
                payload = {
                    "metadata": {
                        "id": document,
                        "language-region": "test",
                        "date": "12",
                        "time": "12",
                    },
                    "token": [
                        {
                            "id": "t1",
                            "form": "nam",
                            "norm": "nam",
                            "lemma": "nëmen",
                            "inflClass": "st4",
                            "grapho": "",
                            "infl": "Ind.Past.Sg.3",
                            "pos_hits": "VVFIN",
                        }
                    ],
                }
                (corpus / f"{document}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            result = extract_mhg(str(corpus), str(Path(tmp) / "mhg.csv"))

        self.assertEqual(set(result["document_id"]), {"MHG:M001", "MHG:M002"})
        self.assertEqual(set(result["token_id"]), {"t1"})
        self.assertEqual(result["observation_id"].nunique(), 2)
        self.assertNotIn("id", result.columns)

    def test_enhg_reads_document_id_from_text_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "enhg"
            source = corpus / "ref-mlu"
            source.mkdir(parents=True)
            template = """<text id="{document}">
              <header>corpus: ReF.MLU\nlanguage-region: test\ndate: 1500\ntime: 16</header>
              <token id="t1"><tok_dipl utf="nam"/><tok_anno utf="nam">
                <posLemma tag="VV"/><pos tag="VVFIN"/><lemma tag="nehmen"/>
                <morph tag="3.Sg.Prät.Ind.St"/>
              </tok_anno></token>
            </text>"""
            for document in ("F001", "F002"):
                (source / f"{document}.xml").write_text(
                    template.format(document=document), encoding="utf-8"
                )

            result = extract_enhg(str(corpus), str(Path(tmp) / "enhg.csv"))

        self.assertEqual(set(result["document_id"]), {"ENHG:F001", "ENHG:F002"})
        self.assertEqual(set(result["token_id"]), {"t1"})
        self.assertEqual(result["observation_id"].nunique(), 2)
        self.assertNotIn("id", result.columns)


class TestLanguageSpecificLemmaJoin(unittest.TestCase):
    @staticmethod
    def corpus_rows(corpus):
        prefix = "MHG" if corpus == "MHG" else "ENHG"
        return pd.DataFrame(
            {
                "lemma": ["shared", "known", "unmapped"],
                "document_id": [f"{prefix}:D1"] * 3,
                "token_id": ["t1", "t2", "t3"],
                "observation_id": [
                    f"{prefix}:D1:t1",
                    f"{prefix}:D1:t2",
                    f"{prefix}:D1:t3",
                ],
            }
        )

    def test_join_is_many_to_one_and_preserves_unmatched_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mhg = self.corpus_rows("MHG")
            enhg = self.corpus_rows("ENHG")
            mapping = pd.DataFrame(
                {
                    "corpus": ["MHG", "ENHG", "MHG", "ENHG"],
                    "lemma": ["shared", "shared", "known", "known"],
                    "lemma_id": [10, 20, 11, 21],
                }
            )
            mhg.to_csv(tmp / "mhg.csv", index=False)
            enhg.to_csv(tmp / "enhg.csv", index=False)
            mapping.to_csv(tmp / "mapping.csv", index=False)

            combined = combine_corpus_data(
                tmp / "enhg.csv",
                tmp / "mhg.csv",
                tmp / "mapping.csv",
                tmp / "combined.csv",
            )

        self.assertEqual(len(combined), len(mhg) + len(enhg))
        shared = combined[combined["lemma"] == "shared"].set_index("corpus")
        self.assertEqual(int(shared.loc["MHG", "lemma_id"]), 10)
        self.assertEqual(int(shared.loc["ENHG", "lemma_id"]), 20)
        self.assertEqual(int(combined["lemma_id"].notna().sum()), 4)
        self.assertEqual(int(combined["lemma_id"].isna().sum()), 2)
        self.assertEqual(combined["observation_id"].nunique(), len(combined))
        self.assertEqual(combined.attrs["join_report"]["MHG"]["unmapped"], 1)
        self.assertEqual(combined.attrs["join_report"]["ENHG"]["unmapped"], 1)

    def test_mapping_key_must_be_unique(self):
        mapping = pd.DataFrame(
            {
                "corpus": ["MHG", "MHG"],
                "lemma": ["same", "same"],
                "lemma_id": [1, 2],
            }
        )
        with self.assertRaisesRegex(ValueError, "not unique"):
            validate_lemma_mapping(mapping)

    def test_mapping_artifact_retains_graph_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lemma_id.csv"
            write_output(
                path,
                {("MHG", "same"): 1, ("ENHG", "same"): 2},
            )
            mapping = pd.read_csv(path)

        self.assertEqual(list(mapping.columns), ["corpus", "lemma", "lemma_id"])
        self.assertEqual(len(mapping), 2)
        validate_lemma_mapping(mapping)


def paired_long_rows():
    rows = []
    tokens = [
        ("MHG:M001:t1", "MHG:M001", "t1", "MHG"),
        ("MHG:M001:t2", "MHG:M001", "t2", "MHG"),
        ("ENHG:F001:t1", "ENHG:F001", "t1", "ENHG"),
    ]
    for observation_id, document_id, token_id, corpus in tokens:
        common = {
            "observation_id": observation_id,
            "document_id": document_id,
            "token_id": token_id,
            "lemma_id": 1,
            "lemma": "same",
            "date": 1500,
            "variety": "Upper German",
            "std_infl": "PastSg",
            "corpus": corpus,
        }
        rows.append({**common, "marking_type": "vowel_bipartite", "has_levelled": 0})
        rows.append({**common, "marking_type": "consonant_bipartite", "has_levelled": 1})
    return pd.DataFrame(rows)


class TestTokenLevelModelingAndPairs(unittest.TestCase):
    def test_pairs_use_collision_free_observation_identity(self):
        pairs = build_paired_cells(paired_long_rows())

        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs["observation_id"].nunique(), 3)
        self.assertEqual(int((pairs["token_id"] == "t1").sum()), 2)
        self.assertEqual(int((pairs["document_id"] == "MHG:M001").sum()), 2)

    def test_pair_builder_rejects_ambiguous_channel_rows(self):
        long = paired_long_rows()
        duplicate = long[long["marking_type"] == "vowel_bipartite"].iloc[[0]]
        with self.assertRaisesRegex(ValueError, "multiple rows for one token"):
            build_paired_cells(pd.concat([long, duplicate], ignore_index=True))

    def test_reshape_retains_repeated_lemma_slot_tokens_in_one_document(self):
        rows = []
        for token_id in ("t1", "t2"):
            rows.append(
                {
                    "lemma_id": 1,
                    "lemma": "same",
                    "std_infl": "PastSg",
                    "form_freq_per_1000": 0.5,
                    "lemma_freq_per_1000": 1.0,
                    "variety": "Upper German",
                    "vowel_alternation_pres": "a ~ e",
                    "vowel_alternation_past": "a ~ u",
                    "cons_alternation_pres": "t ~ d",
                    "cons_alternation_past": "t ~ d",
                    "is_leveled_vowel_pres": 0,
                    "is_leveled_vowel_past": 0,
                    "is_leveled_cons_pres": 1,
                    "is_leveled_cons_past": 0,
                    "is_bipartite": 1,
                    "document_id": "MHG:M001",
                    "token_id": token_id,
                    "observation_id": f"MHG:M001:{token_id}",
                    "date": 1200,
                    "corpus": "MHG",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coded.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            long = reshape(path)

        self.assertEqual(len(long), 4)
        self.assertEqual(long["observation_id"].nunique(), 2)
        self.assertEqual(
            len(long.drop_duplicates(["observation_id", "marking_type"])), 4
        )


if __name__ == "__main__":
    unittest.main()
