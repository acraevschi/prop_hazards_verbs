import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from data.corpus_approach_coding import (
    clean_form,
    extract_root_structure,
    build_lemma_index,
    are_cons_equivalent,
    are_vowels_equivalent,
    load_sound_changes,
    standardize_infl,
    PRINCIPAL_PART_TO_INFL,
    SLOTS_WITH_ENDING,
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

    def test_uber_prefix_stripping(self):
        # Diacritic-normalized prefixes like 'über' must match cleanly
        self.assertEqual(extract_root_structure(None, "überwant", corpus="MHG"), ("a", "n"))
        self.assertEqual(extract_root_structure(None, "überlas", corpus="MHG"), ("a", "s"))
        self.assertEqual(extract_root_structure(None, "überzogen", corpus="MHG"), ("o", "g"))

    def test_root_eating_protection(self):
        # Roots starting with 'be-', 'er-', 'geb-' must NOT have their root onset/nucleus eaten
        self.assertEqual(extract_root_structure(None, "beiz", corpus="MHG"), ("ei", "z"))
        self.assertEqual(extract_root_structure(None, "beizen", corpus="MHG"), ("ei", "z"))
        self.assertEqual(extract_root_structure(None, "erbeiten", corpus="MHG"), ("ei", "t"))
        self.assertEqual(extract_root_structure(None, "geben", corpus="MHG"), ("e", "b"))
        self.assertEqual(extract_root_structure(None, "gegeben", corpus="MHG"), ("e", "b"))
        self.assertEqual(extract_root_structure(None, "vergeben", corpus="MHG"), ("e", "b"))


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

    def test_past_subjunctive_goes_with_the_plural(self):
        """
        The past subjunctive is built on the past plural stem (hulfen -> hülfe),
        so it belongs to principal part 3 with the plural, whatever its own
        number is. map_category in data/normalize_data.py already does this;
        standardize_infl is the fallback and must not disagree.
        """
        for label in ("Subj.Past.Sg.3", "Subj.Past.Sg.1", "3.Sg.Prät.Konj",
                      "1.Sg.Prät.Konj", "(Subj).Past.Sg.3"):
            self.assertEqual(standardize_infl(label), "PastPl", label)
        # The indicative singular is unaffected.
        for label in ("Ind.Past.Sg.3", "3.Sg.Prät.Ind", "*.Past.Sg.3"):
            self.assertEqual(standardize_infl(label), "PastSg", label)

    def test_agrees_with_principal_part_table(self):
        """PRINCIPAL_PART_TO_INFL must cover every part map_category assigns."""
        self.assertEqual(
            set(PRINCIPAL_PART_TO_INFL), {1, 2, 3, 4}
        )
        self.assertEqual(PRINCIPAL_PART_TO_INFL[2], "PastSg")
        self.assertEqual(PRINCIPAL_PART_TO_INFL[3], "PastPl")


class TestDSUAlgorithm(unittest.TestCase):
    def test_disjoint_set_union(self):
        dsu = DSU()
        dsu.union(("ENHG", "sehen"), ("ENHG", "ansehen"))
        dsu.union(("MHG", "sehn"), ("MHG", "ansehn"))
        dsu.union(("ENHG", "sehen"), ("MHG", "sehn"))

        self.assertEqual(dsu.find(("ENHG", "ansehen")), dsu.find(("MHG", "ansehn")))


# A miniature stand-in for the corpus lemma table. ReM marks a prefixed lemma
# with a hyphen and leaves an unprefixed one bare, and that contrast is the
# whole signal build_lemma_index reads.
LEMMA_ROWS = pd.DataFrame(
    {
        "lemma_id": [
            "122", "86", "104", "298", "298", "275", "9", "88", "17", "325", "51",
        ],
        "lemma": [
            "ge-winnen",     # ge- IS a prefix
            "g\u00ebben",        # ge- is NOT a prefix
            "ezzen",         # vowel-initial root
            "b\u00eezen",        # one lemma_id, two lemma strings
            "en-b\u00eezen",     # ... and the second one declares en-
            "\u00fcber-winten",
            "st\u00e2n",
            "ent-trinnen",
            "zi\u00e8hen",
            "ge-w\u00ebsen",
            "bergen",
        ],
        "corpus": ["MHG"] * 11,
    }
)


class TestLemmaIndex(unittest.TestCase):
    def setUp(self):
        self.index = build_lemma_index(LEMMA_ROWS)

    def test_hyphen_marks_a_prefix_and_bare_lemma_does_not(self):
        # The pair that no phonotactic rule can separate: ge+winnen against
        # g(e)+eben. Only the hyphen tells them apart.
        self.assertEqual(self.index["122"]["onsets"], {"w"})
        self.assertIn("ge", self.index["122"]["prefixes"])
        self.assertEqual(self.index["86"]["onsets"], {"g"})
        self.assertEqual(self.index["86"]["prefixes"], set())

    def test_vowel_initial_root_is_recorded_as_such(self):
        self.assertEqual(self.index["104"]["onsets"], {""})

    def test_prefixes_union_across_a_lemma_id(self):
        # b\u00eezen and en-b\u00eezen are the same verb; the declared prefix from
        # either string has to be available to both.
        self.assertEqual(self.index["298"]["onsets"], {"b"})
        self.assertIn("en", self.index["298"]["prefixes"])

    def test_prefix_is_cleaned_like_the_forms(self):
        # PREFIXES holds "\u00fcber" while forms have been reduced to "uber".
        self.assertIn("uber", self.index["275"]["prefixes"])

    def test_enhg_lemmas_are_ignored(self):
        # ReF lemmas are modern infinitives with unmarked prefixes; reading
        # "verlieren" as a root would record the onset v instead of l.
        rows = pd.DataFrame(
            {"lemma_id": ["7"], "lemma": ["verlieren"], "corpus": ["ENHG"]}
        )
        self.assertEqual(build_lemma_index(rows), {})

    def test_missing_frame_is_tolerated(self):
        self.assertEqual(build_lemma_index(None), {})
        self.assertEqual(build_lemma_index(pd.DataFrame({"a": [1]})), {})


class TestLemmaGuidedStripping(unittest.TestCase):
    """
    The pair the phonotactic guards cannot do.

    ge-winnen and g\u00ebben are the same string shape: prefix-looking ge, then a
    single consonant, then the ending. Stripping ge from both gives ben and wan.
    Anything that gets one right by shape alone gets the other wrong, which is
    how gewan came to be parsed as the root wan with the coda w and handed
    ge-winnen a consonant alternation it has never had.
    """

    def setUp(self):
        self.index = build_lemma_index(LEMMA_ROWS)

    def strip(self, form, lemma_id, std_infl):
        return extract_root_structure(
            None,
            form,
            corpus="MHG",
            lemma_id=lemma_id,
            std_infl=std_infl,
            lemma_index=self.index,
        )

    def test_prefix_comes_off_when_the_lemma_declares_it(self):
        # ge-winnen ~ gewan ~ gewunnen: ablaut i ~ a ~ u, coda n throughout.
        self.assertEqual(self.strip("gewinnen", "122", "Pres"), ("i", "n"))
        self.assertEqual(self.strip("gewan", "122", "PastSg"), ("a", "n"))
        self.assertEqual(self.strip("gewunnen", "122", "PastPl"), ("u", "n"))

    def test_prefix_stays_on_when_the_lemma_does_not_declare_it(self):
        # g\u00ebben ~ gap ~ g\u00e2ben: the b is the root coda, not a prefix boundary.
        self.assertEqual(self.strip("geben", "86", "Pres"), ("e", "b"))
        self.assertEqual(self.strip("gegeben", "86", "Ppl"), ("e", "b"))
        self.assertEqual(self.strip("gap", "86", "PastSg"), ("a", "p"))
        self.assertEqual(self.strip("gaben", "86", "PastPl"), ("a", "b"))

    def test_vowel_initial_roots_survive(self):
        # ezzen ~ az ~ \u00e2zen. The phonotactic fallback rejects every
        # vowel-initial remainder and so loses these; the lemma path keeps them.
        self.assertEqual(self.strip("geaz", "104", "PastSg"), ("a", "z"))
        self.assertEqual(self.strip("ezzen", "104", "Pres"), ("e", "z"))

    def test_class_one_diphthong_is_not_eaten(self):
        # be- off beiz leaves iz, collapsing the Class I ei ~ i ablaut.
        self.assertEqual(self.strip("beiz", "298", "PastSg"), ("ei", "z"))
        # en-b\u00eezen declares en-, so enbeiz reduces to the same root.
        self.assertEqual(self.strip("enbeiz", "298", "PastSg"), ("ei", "z"))

    def test_uber_prefix_matches_after_diacritic_stripping(self):
        self.assertEqual(self.strip("\u00fcberwant", "275", "PastSg"), ("a", "n"))
        self.assertEqual(self.strip("\u00fcberwinden", "275", "Pres"), ("i", "n"))

    def test_contracted_vowel_final_roots(self):
        # st\u00e2n is vowel-final, so be-/ver- must come off and the -n must go.
        self.assertEqual(self.strip("bestan", "9", "Pres"), ("a", ""))
        self.assertEqual(self.strip("verstan", "9", "Pres"), ("a", ""))

    def test_degemination_at_the_prefix_boundary(self):
        # ent- plus trinnen is written entran, so the candidate ran has to be
        # read against the root onset tr.
        self.assertEqual(self.strip("entran", "88", "PastSg"), ("a", "n"))
        self.assertEqual(self.strip("entrinnen", "88", "Pres"), ("i", "n"))


class TestInflectionAwareEndings(unittest.TestCase):
    """
    A coda of nothing but -n is an ending in the present, the past plural and
    the participle, and root material in the past singular.
    """

    def test_slot_table(self):
        self.assertEqual(SLOTS_WITH_ENDING, {"Pres", "PastPl", "Ppl"})

    def test_past_singular_keeps_its_root_n(self):
        self.assertEqual(
            extract_root_structure(None, "wan", corpus="MHG", std_infl="PastSg"),
            ("a", "n"),
        )
        self.assertEqual(
            extract_root_structure(None, "began", corpus="MHG", std_infl="PastSg"),
            ("a", "n"),
        )

    def test_other_slots_drop_the_ending(self):
        # gan and schrien are genuinely vowel-final roots.
        self.assertEqual(
            extract_root_structure(None, "gan", corpus="MHG", std_infl="Pres"),
            ("a", ""),
        )
        self.assertEqual(
            extract_root_structure(None, "schrien", corpus="MHG", std_infl="Pres"),
            ("ie", ""),
        )

    def test_absent_slot_behaves_like_an_ending_bearing_one(self):
        self.assertEqual(
            extract_root_structure(None, "schrien", corpus="MHG"), ("ie", "")
        )

    def test_a_coda_that_is_not_bare_n_is_never_emptied(self):
        # Only a bare -n is ever deleted outright. The t of gat and the p of
        # gap are the whole coda and are not the infinitive marker, so they
        # stay, and the verb does not silently become vowel-final.
        self.assertEqual(
            extract_root_structure(None, "gat", corpus="MHG", std_infl="Pres"),
            ("a", "t"),
        )
        self.assertEqual(
            extract_root_structure(None, "gap", corpus="MHG", std_infl="PastSg"),
            ("a", "p"),
        )
        # A longer coda is still shortened by its ending: the -t of gilt is the
        # 3rd singular, leaving the root's l.
        self.assertEqual(
            extract_root_structure(None, "gilt", corpus="MHG", std_infl="Pres"),
            ("i", "l"),
        )


class TestQuDigraph(unittest.TestCase):
    def test_qu_is_resolved_before_the_vowel_regex_runs(self):
        # quam is the past singular of komen. Reading the u of qu as nuclear
        # gave the anchor "ua" and made the whole verb uninformative.
        self.assertEqual(clean_form("quam"), "kwam")
        self.assertEqual(
            extract_root_structure(None, "quam", corpus="MHG", std_infl="PastSg"),
            ("a", "m"),
        )
        self.assertEqual(
            extract_root_structure(None, "qu\u00ebden", corpus="MHG", std_infl="Pres"),
            ("e", "d"),
        )

    def test_uo_diphthong_is_untouched(self):
        self.assertEqual(
            extract_root_structure(None, "tuon", corpus="MHG", std_infl="Pres"),
            ("uo", ""),
        )
        self.assertEqual(
            extract_root_structure(None, "sluoc", corpus="MHG", std_infl="PastSg"),
            ("uo", "c"),
        )


if __name__ == "__main__":
    unittest.main()
