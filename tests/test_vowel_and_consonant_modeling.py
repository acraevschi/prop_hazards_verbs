import unittest
import os
import sys
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.marking_type_summary import reshape
from analysis.consonant_analysis import (
    load_and_reshape,
    analyze_marking_rates,
    analyze_consonant_lemmas,
    compute_mechanism_summary,
    compute_statistical_contrasts,
    MORPHOLOGICAL_GW_LEMMAS,
    ORTHOGRAPHIC_CODA_LEMMAS,
)


class TestVowelModelingData(unittest.TestCase):
    """Verifies that the dataset prepared for brms models conforms to the Vowel-Only Option A specification."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../analysis/data_for_analysis.csv"))
        cls.df = pd.read_csv(cls.data_path)

    def test_vowel_only_marking_types(self):
        marking_types = set(self.df["marking_type"].unique())
        self.assertEqual(marking_types, {"vowel_unipartite", "vowel_bipartite"})
        self.assertNotIn("consonant_bipartite", marking_types)

    def test_no_missing_critical_fields(self):
        critical_cols = ["lemma", "lemma_std", "date", "marking_type", "has_levelled", "variety", "std_infl"]
        for col in critical_cols:
            self.assertIn(col, self.df.columns)
            self.assertEqual(self.df[col].isna().sum(), 0, f"Column {col} has unexpected NA values")

    def test_binary_leveling_outcomes(self):
        values = set(self.df["has_levelled"].unique())
        self.assertTrue(values.issubset({0, 1}))


class TestConsonantAnalysis(unittest.TestCase):
    """Verifies the dedicated consonant channel analysis."""

    @classmethod
    def setUpClass(cls):
        cls.coded_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/coded_output.csv"))
        cls.long_df, cls.raw_df = load_and_reshape(cls.coded_path)
        cls.rates_df = analyze_marking_rates(cls.long_df)
        cls.lemma_df = analyze_consonant_lemmas(cls.long_df, cls.raw_df)
        cls.mech_df = compute_mechanism_summary(cls.lemma_df)
        cls.contrasts = compute_statistical_contrasts(cls.long_df, cls.lemma_df)

    def test_consonant_rate_magnitude(self):
        cons_rate = self.rates_df.loc[self.rates_df["marking_type"] == "consonant_bipartite", "rate_pct"].values[0]
        vowel_bi_rate = self.rates_df.loc[self.rates_df["marking_type"] == "vowel_bipartite", "rate_pct"].values[0]
        vowel_uni_rate = self.rates_df.loc[self.rates_df["marking_type"] == "vowel_unipartite", "rate_pct"].values[0]

        # Consonant rate should be ~8% and higher than vowel rates
        self.assertGreater(cons_rate, 7.0)
        self.assertLess(cons_rate, 10.0)
        self.assertGreater(cons_rate, vowel_bi_rate)
        self.assertGreater(cons_rate, vowel_uni_rate)

    def test_mechanism_classification(self):
        categories = set(self.mech_df["category"].unique())
        self.assertIn("Morphological GW", categories)
        self.assertIn("Orthographic / Devoicing", categories)

        # Check that top verbs are correctly categorized
        self.assertIn(17, MORPHOLOGICAL_GW_LEMMAS) # ziehen
        self.assertIn(216, MORPHOLOGICAL_GW_LEMMAS) # verlieren
        self.assertIn(95, ORTHOGRAPHIC_CODA_LEMMAS) # lîden
        self.assertIn(145, ORTHOGRAPHIC_CODA_LEMMAS) # lîhen

    def test_statistical_contrast_odds_ratio(self):
        or_cons_vs_vowel_bi = self.contrasts["Cons_All_vs_Vowel_Bi"]["odds_ratio"]
        self.assertGreater(or_cons_vs_vowel_bi, 5.0)


if __name__ == "__main__":
    unittest.main()
