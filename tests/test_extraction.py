import unittest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.corpus_approach_coding import (
    clean_form,
    extract_root_structure,
    are_cons_equivalent,
    are_vowels_equivalent,
    load_sound_changes,
    standardize_infl,
)
from data.lemmas.enhg_mhg_mapping import DSU


class TestOrthographicCleaning(unittest.TestCase):
    def test_strip_editorial_brackets(self):
        # Brackets and damaged/editorial reconstructions inside brackets are stripped
        self.assertEqual(clean_form("sn[unclear]iden"), "sniden")
        self.assertEqual(clean_form("ge[ge]ben"), "geben")
        self.assertEqual(clean_form("sn[?]iden"), "sniden")

    def test_vowel_normalization(self):
        # Long vowels with circumflex/macron mapped to base vowels
        self.assertEqual(clean_form("snîden"), "sniden")
        self.assertEqual(clean_form("lâzen"), "lazen")
        self.assertEqual(clean_form("rîten"), "riten")
        self.assertEqual(clean_form("bûgen"), "bugen")

    def test_consonant_normalization(self):
        # Historical sibilants and ligatures mapped to standard characters
        self.assertEqual(clean_form("verlôs"), "verlos")
        self.assertEqual(clean_form("verloſ"), "verlos")
        self.assertEqual(clean_form("verloß"), "verlos")
        self.assertEqual(clean_form("verloʒ"), "verlos")


class TestPhonotacticRootExtraction(unittest.TestCase):
    def test_verner_alternation_class_1(self):
        # snîden -> sniden / gesniten
        v_inf, c_inf = extract_root_structure(None, "snîden", corpus="MHG")
        v_pp, c_pp = extract_root_structure(None, "gesniten", corpus="MHG")
        self.assertEqual(v_inf, "i")
        self.assertEqual(c_inf, "d")
        self.assertEqual(v_pp, "i")
        self.assertEqual(c_pp, "t")

    def test_verner_alternation_class_2(self):
        # verliesen -> verlos / verlurn
        v_inf, c_inf = extract_root_structure(None, "verliesen", corpus="MHG")
        v_sg, c_sg = extract_root_structure(None, "verlôs", corpus="MHG")
        self.assertEqual(v_inf, "ie")
        self.assertEqual(c_inf, "s")
        self.assertEqual(v_sg, "o")
        self.assertEqual(c_sg, "s")

    def test_illegal_onset_guard(self):
        # In 'gelten', stripping 'ge-' would yield illegal onset 'lt' -> strip is rejected
        v, c = extract_root_structure(None, "gelten", corpus="MHG")
        self.assertEqual(v, "e")
        self.assertIsNotNone(c)

    def test_geminate_onset_guard(self):
        # In 'gessen', stripping 'ge-' would yield geminate 'ss' -> strip is rejected
        v, c = extract_root_structure(None, "gessen", corpus="MHG")
        self.assertEqual(v, "e")
        self.assertEqual(c, "s")

    def test_structural_coda_protection(self):
        # geschehen -> ge- is stripped, root 'scheh' yields vowel 'e' and coda 'h' (not stripped as -en)
        v, c = extract_root_structure(None, "geschehen", corpus="MHG")
        self.assertEqual(v, "e")
        self.assertEqual(c, "h")


class TestEquivalenceSets(unittest.TestCase):
    def test_consonant_devoicing_and_spelling(self):
        self.assertTrue(are_cons_equivalent("p", "b"))
        self.assertTrue(are_cons_equivalent("v", "f"))
        self.assertTrue(are_cons_equivalent("u", "v"))
        self.assertTrue(are_cons_equivalent("s", "z"))
        self.assertTrue(are_cons_equivalent("s", "ß"))
        self.assertTrue(are_cons_equivalent("h", "χ"))

    def test_grammatischer_wechsel_distinction(self):
        # GW pairs must NOT be marked equivalent
        self.assertFalse(are_cons_equivalent("d", "t"))
        self.assertFalse(are_cons_equivalent("s", "r"))
        self.assertFalse(are_cons_equivalent("h", "g"))
        self.assertFalse(are_cons_equivalent("v", "b"))


class TestSoundChanges(unittest.TestCase):
    def setUp(self):
        self.sc_dict = load_sound_changes("data/vowel_changes.csv")

    def test_sound_change_filtering(self):
        # Upper German uo -> u
        self.assertTrue(are_vowels_equivalent("uo", "u", "Upper German", self.sc_dict))
        # Central German i -> ei
        self.assertTrue(are_vowels_equivalent("i", "ei", "Central German", self.sc_dict))


class TestInflectionStandardization(unittest.TestCase):
    def test_slots(self):
        self.assertEqual(standardize_infl("Ind.Pres.1.Sg"), "Pres")
        self.assertEqual(standardize_infl("Ind.Past.1.Sg"), "PastSg")
        self.assertEqual(standardize_infl("Ind.Past.3.Sg"), "PastSg")
        self.assertEqual(standardize_infl("Ind.Past.2.Sg"), "PastPl")  # 2nd sg patterns with plural in MHG
        self.assertEqual(standardize_infl("Ind.Past.1.Pl"), "PastPl")
        self.assertEqual(standardize_infl("Ind.Past.3.Pl"), "PastPl")


class TestDSUAlgorithm(unittest.TestCase):
    def test_disjoint_set_union(self):
        dsu = DSU()
        dsu.union(("ENHG", "sehen"), ("ENHG", "ansehen"))
        dsu.union(("MHG", "sehn"), ("MHG", "ansehn"))
        dsu.union(("ENHG", "sehen"), ("MHG", "sehn"))

        self.assertEqual(dsu.find(("ENHG", "ansehen")), dsu.find(("MHG", "ansehn")))


if __name__ == "__main__":
    unittest.main()
