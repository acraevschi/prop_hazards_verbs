import unittest
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
        # Historical sibilants, ligatures, and scribal macrons mapped to standard characters
        self.assertEqual(clean_form("verlôs"), "verlos")
        self.assertEqual(clean_form("verloſ"), "verlos")
        self.assertEqual(clean_form("verloß"), "verlos")
        self.assertEqual(clean_form("verloʒ"), "verlos")
        self.assertEqual(clean_form("kôn̄en"), "konen")


class TestAblautClassesExtraction(unittest.TestCase):
    """
    Validates root extraction across all seven historical High German Ablaut classes.
    """

    def test_class_1_ablaut(self):
        # Class I: î - ei - i - i (e.g. rîten / reit / riten / geriten)
        self.assertEqual(extract_root_structure(None, "rîten", corpus="MHG"), ("i", "t"))
        self.assertEqual(extract_root_structure(None, "reit", corpus="MHG"), ("ei", "t"))
        self.assertEqual(extract_root_structure(None, "riten", corpus="MHG"), ("i", "t"))
        self.assertEqual(extract_root_structure(None, "geriten", corpus="MHG"), ("i", "t"))

    def test_class_2_ablaut(self):
        # Class II: ie - ou - u - o (e.g. bîegen / bouc / bugen / gebogen)
        self.assertEqual(extract_root_structure(None, "bîegen", corpus="MHG"), ("ie", "g"))
        self.assertEqual(extract_root_structure(None, "bouc", corpus="MHG"), ("ou", "c"))
        self.assertEqual(extract_root_structure(None, "bugen", corpus="MHG"), ("u", "g"))
        self.assertEqual(extract_root_structure(None, "gebogen", corpus="MHG"), ("o", "g"))

    def test_class_3_ablaut(self):
        # Class III: e/i - a - u - o/u (e.g. helfen / half / hulfen / geholfen)
        self.assertEqual(extract_root_structure(None, "helfen", corpus="MHG"), ("e", "lf"))
        self.assertEqual(extract_root_structure(None, "half", corpus="MHG"), ("a", "lf"))
        self.assertEqual(extract_root_structure(None, "hulfen", corpus="MHG"), ("u", "lf"))
        self.assertEqual(extract_root_structure(None, "geholfen", corpus="MHG"), ("o", "lf"))

    def test_class_4_ablaut(self):
        # Class IV: e - a - â - o (e.g. nëmen / nam / nâmen / genomen)
        self.assertEqual(extract_root_structure(None, "nëmen", corpus="MHG"), ("e", "m"))
        self.assertEqual(extract_root_structure(None, "nam", corpus="MHG"), ("a", "m"))
        self.assertEqual(extract_root_structure(None, "nâmen", corpus="MHG"), ("a", "m"))
        self.assertEqual(extract_root_structure(None, "genomen", corpus="MHG"), ("o", "m"))

    def test_class_5_ablaut(self):
        # Class V: e - a - â - e (e.g. gâben / gap, sëhen / sach / sâhen / gesehen)
        self.assertEqual(extract_root_structure(None, "gap", corpus="MHG"), ("a", "p"))
        self.assertEqual(extract_root_structure(None, "gâben", corpus="MHG"), ("a", "b"))
        self.assertEqual(extract_root_structure(None, "sëhen", corpus="MHG"), ("e", "h"))
        self.assertEqual(extract_root_structure(None, "sach", corpus="MHG"), ("a", "χ"))
        self.assertEqual(extract_root_structure(None, "sâhen", corpus="MHG"), ("a", "h"))
        self.assertEqual(extract_root_structure(None, "gesehen", corpus="MHG"), ("e", "h"))

    def test_class_6_ablaut(self):
        # Class VI: a - uo - uo - a (e.g. tragen / truoc / truogen / getragen)
        self.assertEqual(extract_root_structure(None, "tragen", corpus="MHG"), ("a", "g"))
        self.assertEqual(extract_root_structure(None, "truoc", corpus="MHG"), ("uo", "c"))
        self.assertEqual(extract_root_structure(None, "truogen", corpus="MHG"), ("uo", "g"))
        self.assertEqual(extract_root_structure(None, "getragen", corpus="MHG"), ("a", "g"))
        # graben / gruop / gruoben / gegraben
        self.assertEqual(extract_root_structure(None, "graben", corpus="MHG"), ("a", "b"))
        self.assertEqual(extract_root_structure(None, "gruop", corpus="MHG"), ("uo", "p"))
        self.assertEqual(extract_root_structure(None, "gruoben", corpus="MHG"), ("uo", "b"))
        self.assertEqual(extract_root_structure(None, "gegraben", corpus="MHG"), ("a", "b"))

    def test_class_7_ablaut(self):
        # Class VII: Reduplicating / Variable (e.g. lâzen / liez / gelâzen, fallen / fiel / gefallen)
        self.assertEqual(extract_root_structure(None, "lâzen", corpus="MHG"), ("a", "z"))
        self.assertEqual(extract_root_structure(None, "liez", corpus="MHG"), ("ie", "z"))
        self.assertEqual(extract_root_structure(None, "liezen", corpus="MHG"), ("ie", "z"))
        self.assertEqual(extract_root_structure(None, "gelâzen", corpus="MHG"), ("a", "z"))
        self.assertEqual(extract_root_structure(None, "fallen", corpus="MHG"), ("a", "l"))
        self.assertEqual(extract_root_structure(None, "fiel", corpus="MHG"), ("ie", "l"))
        self.assertEqual(extract_root_structure(None, "gefallen", corpus="MHG"), ("a", "l"))


class TestGrammatischerWechselPairs(unittest.TestCase):
    """
    Validates extraction across all four historical Verner's Law consonant alternations.
    """

    def test_gw_d_t(self):
        # snîden -> sneit / sniten / gesniten
        v_inf, c_inf = extract_root_structure(None, "snîden", corpus="MHG")
        v_sg, c_sg = extract_root_structure(None, "sneit", corpus="MHG")
        v_pl, c_pl = extract_root_structure(None, "sniten", corpus="MHG")
        v_pp, c_pp = extract_root_structure(None, "gesniten", corpus="MHG")
        self.assertEqual((v_inf, c_inf), ("i", "d"))
        self.assertEqual((v_sg, c_sg), ("ei", "t"))
        self.assertEqual((v_pl, c_pl), ("i", "t"))
        self.assertEqual((v_pp, c_pp), ("i", "t"))
        self.assertFalse(are_cons_equivalent(c_inf, c_pl))

    def test_gw_h_g(self):
        # zîhen -> zêch / zigen / gezogen
        v_inf, c_inf = extract_root_structure(None, "zîhen", corpus="MHG")
        v_sg, c_sg = extract_root_structure(None, "zêch", corpus="MHG")
        v_pl, c_pl = extract_root_structure(None, "zigen", corpus="MHG")
        v_pp, c_pp = extract_root_structure(None, "gezogen", corpus="MHG")
        self.assertEqual((v_inf, c_inf), ("i", "h"))
        self.assertEqual((v_sg, c_sg), ("e", "χ"))
        self.assertEqual((v_pl, c_pl), ("i", "g"))
        self.assertEqual((v_pp, c_pp), ("o", "g"))
        self.assertFalse(are_cons_equivalent(c_inf, c_pl))

    def test_gw_s_r(self):
        # verliesen -> verlôs / verlurn / verlorn
        v_inf, c_inf = extract_root_structure(None, "verliesen", corpus="MHG")
        v_sg, c_sg = extract_root_structure(None, "verlôs", corpus="MHG")
        v_pl, c_pl = extract_root_structure(None, "verlurn", corpus="MHG")
        v_pp, c_pp = extract_root_structure(None, "verlorn", corpus="MHG")
        self.assertEqual((v_inf, c_inf), ("ie", "s"))
        self.assertEqual((v_sg, c_sg), ("o", "s"))
        self.assertEqual((v_pl, c_pl), ("u", "r"))
        self.assertEqual((v_pp, c_pp), ("o", "r"))
        self.assertFalse(are_cons_equivalent(c_inf, c_pl))

    def test_gw_v_b(self):
        # heven -> huop / huoben / gehaben
        v_inf, c_inf = extract_root_structure(None, "heven", corpus="MHG")
        v_sg, c_sg = extract_root_structure(None, "huop", corpus="MHG")
        v_pl, c_pl = extract_root_structure(None, "huoben", corpus="MHG")
        v_pp, c_pp = extract_root_structure(None, "gehaben", corpus="MHG")
        self.assertEqual((v_inf, c_inf), ("e", "v"))
        self.assertEqual((v_sg, c_sg), ("uo", "p"))
        self.assertEqual((v_pl, c_pl), ("uo", "b"))
        self.assertEqual((v_pp, c_pp), ("a", "b"))
        self.assertFalse(are_cons_equivalent(c_inf, c_pl))


class TestPhonotacticGuards(unittest.TestCase):
    def test_illegal_onset_guard(self):
        # In 'gelten', stripping 'ge-' would yield illegal onset 'lt' -> strip is rejected
        v, c = extract_root_structure(None, "gelten", corpus="MHG")
        self.assertEqual(v, "e")
        self.assertEqual(c, "l")

        # In 'bergen', stripping 'be-' would yield illegal onset 'rg' -> strip is rejected
        v_b, c_b = extract_root_structure(None, "bergen", corpus="MHG")
        self.assertEqual(v_b, "e")
        self.assertEqual(c_b, "rg")

        # In 'gerben', stripping 'ge-' would yield illegal onset 'rb' -> strip is rejected
        v_g, c_g = extract_root_structure(None, "gerben", corpus="MHG")
        self.assertEqual(v_g, "e")
        self.assertEqual(c_g, "rb")

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

        v_past, c_past = extract_root_structure(None, "geschach", corpus="MHG")
        self.assertEqual(v_past, "a")
        self.assertEqual(c_past, "χ")

        # vowel-final roots
        v_sch, c_sch = extract_root_structure(None, "schrien", corpus="MHG")
        self.assertEqual((v_sch, c_sch), ("ie", ""))

    def test_compound_prefix_stripping(self):
        # Compound verbs with recognized prefixes must strip correctly
        self.assertEqual(extract_root_structure(None, "vertragen", corpus="MHG"), ("a", "g"))
        self.assertEqual(extract_root_structure(None, "begraben", corpus="MHG"), ("a", "b"))
        self.assertEqual(extract_root_structure(None, "zerbrechen", corpus="MHG"), ("e", "χ"))


class TestEquivalenceSets(unittest.TestCase):
    def test_consonant_devoicing_and_spelling(self):
        self.assertTrue(are_cons_equivalent("p", "b"))
        self.assertTrue(are_cons_equivalent("v", "f"))
        self.assertTrue(are_cons_equivalent("u", "v"))
        self.assertTrue(are_cons_equivalent("s", "z"))
        self.assertTrue(are_cons_equivalent("s", "ß"))
        self.assertTrue(are_cons_equivalent("h", "χ"))
        self.assertTrue(are_cons_equivalent("c", "g"))
        self.assertTrue(are_cons_equivalent("c", "k"))

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
        self.assertEqual(standardize_infl("PastParticiple"), "Ppl")


class TestDSUAlgorithm(unittest.TestCase):
    def test_disjoint_set_union(self):
        dsu = DSU()
        dsu.union(("ENHG", "sehen"), ("ENHG", "ansehen"))
        dsu.union(("MHG", "sehn"), ("MHG", "ansehn"))
        dsu.union(("ENHG", "sehen"), ("MHG", "sehn"))

        self.assertEqual(dsu.find(("ENHG", "ansehen")), dsu.find(("MHG", "ansehn")))


if __name__ == "__main__":
    unittest.main()
