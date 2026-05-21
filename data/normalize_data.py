import pandas as pd
import json
import argparse


def parse_category(s):
    s_clean = s.replace("(", "").replace(")", "")
    tokens = s_clean.split(".")
    mood = None
    tense = None
    number = None
    person = None

    for token in tokens:
        if token in ["Ind", "Subj", "Konj"]:
            if token == "Konj":
                mood = "Subj"
            else:
                mood = token
        elif token in ["Pres", "Past", "Präs", "Prät"]:
            if token in ["Pres", "Präs"]:
                tense = "Pres"
            else:
                tense = "Past"
        elif token in ["Sg", "Pl"]:
            number = token
        elif token in ["1", "2", "3"]:
            person = token
        elif token == "*":
            pass

    return mood, tense, number, person


def map_category(s):
    # explicit special cases
    if s == "infinitive":
        return 1
    if s == "past_participle":
        return 4
    if s == "--":
        return None

    mood, tense, number, person = parse_category(s)

    if tense is None:
        return None

    # Present tense (Pres): map all present forms to principal part 1 (infinitive)
    if tense == "Pres":
        return 1

    # Past tense:
    # - Preterite subjunctive (Konj./Subj + Past) -> PP3
    # - Indicative (or unspecified mood treated as Indic):
    #     * Sg + person 1 or 3 -> PP2
    #     * Sg + person 2 -> PP3
    #     * Pl + any person -> PP3
    if tense == "Past":
        if mood == "Subj":
            return 3

        if mood is None or mood == "Ind" or mood == "*":
            if number == "Sg":
                if person in ["1", "3"]:
                    return 2
                elif person == "2":
                    return 3
                else:
                    return None
            elif number == "Pl":
                if person in ["1", "2", "3"]:
                    return 3
                else:
                    return None
            else:
                return None

    return None


def main(args):
    data = pd.read_csv(args.input, encoding="utf-8")

    for i in data["lemma_id"].unique():
        num_rows = len(data[data["lemma_id"] == i])
        if num_rows <= 10:
            data.drop(data[data["lemma_id"] == i].index, inplace=True)

    data.reset_index(drop=True, inplace=True)

    with open(args.dialect_mapping, "r", encoding="utf-8") as f:
        dialect_mapping = json.load(f)
    with open(args.date_mapping, "r", encoding="utf-8") as f:
        date_mapping = json.load(f)

    dialect_dict = {item["original"]: item["normalized"] for item in dialect_mapping}
    data["variety"] = data["language-region"].map(dialect_dict)

    date_dict = {item["original"]: item["normalized"] for item in date_mapping}
    data["date"] = data["date"].map(date_dict)

    data["principal_part"] = data["infl"].apply(map_category)

    data.dropna(subset=["lemma_id", "principal_part", "date", "variety"], inplace=True)
    data.drop(
        ["language-region", "specific_dating", "time", "grapho"],
        axis=1,
        inplace=True,
    )
    data["principal_part"] = data["principal_part"].astype(int)
    data.reset_index(drop=True, inplace=True)

    data.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize corpus data")
    parser.add_argument(
        "--input", default="data/combined_corpus.csv", help="Input CSV file"
    )
    parser.add_argument(
        "--date_mapping",
        default="data/date_mapping.json",
        help="Date mapping JSON file",
    )
    parser.add_argument(
        "--dialect_mapping",
        default="data/dialect_mapping.json",
        help="Dialect mapping JSON file",
    )
    parser.add_argument(
        "--output",
        default="data/combined_normalized_corpus.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()
    main(args)
