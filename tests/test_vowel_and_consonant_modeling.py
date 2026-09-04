import unittest
import os
import sys
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.marking_type_summary import reshape
from analysis.consonant_analysis import (
    load_and_reshape,
    build_paired_cells,
    paired_channel_test,
    paired_lemma_breakdown,
    analyze_marking_rates,
    analyze_consonant_lemmas,
    compute_mechanism_summary,
    compute_statistical_contrasts,
    classify_consonant_lemma,
    MORPHOLOGICAL_CATEGORIES,
    MECHANISM_DEVOICING,
    MECHANISM_VERNER_SG_PL,
    MECHANISM_VERNER_MEDIAL,
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
        # Every paradigm in the consonant channel passed the bipartite shape
        # test, so every category present must be one of the Verner clauses.
        self.assertTrue(categories)
        self.assertTrue(categories.issubset(set(MORPHOLOGICAL_CATEGORIES)))
        # The devoicing shape is excluded upstream; a member here would mean the
        # bipartite rule had changed without this report being updated.
        self.assertNotIn(MECHANISM_DEVOICING, categories)

        # The category is read off the paradigm, through the same clauses the
        # bipartite rule uses, so it cannot disagree with what admitted the lemma.
        def row(pres, sg, pl, gw_ps, gw_sp, gw_ppl=False, ab_ppl=False):
            return {
                "anchor_coda_pres": pres,
                "anchor_coda_pastsg": sg,
                "anchor_coda_pastpl": pl,
                "diff_cons_pres_pastsg": str(gw_ps),
                "diff_cons_pastsg_pastpl": str(gw_sp),
                "diff_cons_pres_pastpl": str(gw_ppl),
                "diff_vowel_pres_pastpl": str(ab_ppl),
            }

        # wesen s ~ s ~ r and quëden t ~ t ~ d: the past plural is the odd cell.
        self.assertEqual(classify_consonant_lemma(row("s", "s", "r", False, True)),
                         MECHANISM_VERNER_SG_PL)
        self.assertEqual(classify_consonant_lemma(row("t", "t", "d", False, True)),
                         MECHANISM_VERNER_SG_PL)
        # snîden d ~ t ~ t: the alternation is medial, so it is Verner, not devoicing.
        self.assertEqual(classify_consonant_lemma(row("d", "t", "t", True, False)),
                         MECHANISM_VERNER_MEDIAL)
        # scheiden d ~ t ~ d: the past singular is the odd cell. The upstream rule
        # keeps this out of the channel, so seeing it here means the rule changed.
        self.assertEqual(classify_consonant_lemma(row("d", "t", "d", True, True)),
                         MECHANISM_DEVOICING)
        self.assertNotIn(MECHANISM_DEVOICING, MORPHOLOGICAL_CATEGORIES)

    def test_statistical_contrast_odds_ratio(self):
        or_cons_vs_vowel_bi = self.contrasts["Cons_All_vs_Vowel_Bi"]["odds_ratio"]
        self.assertGreater(or_cons_vs_vowel_bi, 5.0)


class TestPairedChannelAsymmetry(unittest.TestCase):
    """
    Verifies the within-cell design: the vowel and the consonant row of one
    bipartite cell are a matched pair, and the channel question is answered on
    the discordant pairs rather than by comparing the two channels as if they
    were independent samples.
    """

    @classmethod
    def setUpClass(cls):
        coded = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/coded_output.csv"))
        cls.long_df, _ = load_and_reshape(coded)
        cls.pairs = build_paired_cells(cls.long_df)
        cls.result = paired_channel_test(cls.pairs)
        cls.breakdown = paired_lemma_breakdown(cls.pairs)

    def test_pairs_are_a_subset_of_both_channels(self):
        # Every paired cell must exist on both sides. Only bipartite paradigms
        # have a consonant row, so the pairing cannot exceed either channel.
        n_vowel_bi = len(self.long_df[self.long_df["marking_type"] == "vowel_bipartite"])
        n_cons_bi = len(self.long_df[self.long_df["marking_type"] == "consonant_bipartite"])
        self.assertGreater(len(self.pairs), 0)
        self.assertLessEqual(len(self.pairs), n_cons_bi)
        self.assertLessEqual(len(self.pairs), n_vowel_bi)

    def test_pairing_is_one_to_one(self):
        # A cell must not match twice, or the discordance counts would inflate.
        self.assertEqual(len(self.pairs), len(self.pairs.drop_duplicates()))

    def test_contingency_table_is_complete(self):
        r = self.result
        self.assertEqual(
            r["both"] + r["neither"] + r["cons_only"] + r["vowel_only"], r["n_pairs"]
        )
        self.assertEqual(r["discordant"], r["cons_only"] + r["vowel_only"])

    def test_consonant_gives_way_more_often(self):
        r = self.result
        self.assertGreater(r["cons_only"], r["vowel_only"])
        self.assertGreater(r["point_pct"], 50.0)

    def test_clustered_interval_is_wider_than_the_point(self):
        # The whole reason the interval is bootstrapped over lemmas is that the
        # events are concentrated. It must bracket the point estimate and be
        # visibly wide - an interval as tight as the exact p-value would mean
        # the clustering was not applied.
        r = self.result
        self.assertLessEqual(r["ci_lower_pct"], r["point_pct"])
        self.assertGreaterEqual(r["ci_upper_pct"], r["point_pct"])
        self.assertGreater(r["ci_upper_pct"] - r["ci_lower_pct"], 5.0)

    def test_breakdown_accounts_for_every_discordant_pair(self):
        self.assertEqual(int(self.breakdown["discordant"].sum()), self.result["discordant"])
        self.assertEqual(int(self.breakdown["cons_only"].sum()), self.result["cons_only"])
        self.assertEqual(int(self.breakdown["vowel_only"].sum()), self.result["vowel_only"])


if __name__ == "__main__":
    unittest.main()
