### Link lemmas across MHG and ENHG corpora into shared lemma IDs.

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd

Lang = str  # "ENHG" | "MHG"
Node = Tuple[Lang, str]


class DSU:
    def __init__(self):
        self.parent: Dict[Node, Node] = {}
        self.rank: Dict[Node, int] = {}

    def add(self, x: Node) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Node) -> Node:
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: Node, b: Node) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        ra_r, rb_r = self.rank[ra], self.rank[rb]
        if ra_r < rb_r:
            self.parent[ra] = rb
        elif ra_r > rb_r:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


WS_RE = re.compile(r"\s+")


def norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = WS_RE.sub(" ", s)
    # No case-folding or diacritic changes — keep original spelling
    return s


def read_mapping(path: Path, lang: Lang) -> Dict[str, str]:
    """Read derived->base mapping JSON for one language; ensure base self-maps."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        k2, v2 = norm(k), norm(v)
        if not k2 or not v2:
            continue
        out[k2] = v2
    # Ensure base forms self-map
    for base in set(out.values()):
        out.setdefault(base, base)
    return out


def read_crosslinks(path: Path) -> List[Tuple[str, str]]:
    """
    Read ENHG<->MHG base crosslinks from CSV.
    Returns list of (enhg_base, mhg_base) pairs.
    Column names expected:
      - 'ENHG Lemma'
      - 'MHG Candidates' (semicolon-separated list)
    Extra columns (e.g., 'Link') are ignored.
    """
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Be flexible about header capitalization/spacing
        # Build a mapping from normalized header to real header
        headers = {norm(h).lower(): h for h in reader.fieldnames or []}
        enhg_key = headers.get("enhg lemma")
        mhg_key = headers.get("mhg candidates")
        if not enhg_key or not mhg_key:
            raise ValueError(
                f"CSV headers must include 'ENHG Lemma' and 'MHG Candidates'; found: {reader.fieldnames}"
            )
        for row in reader:
            enhg = norm(row.get(enhg_key, ""))
            mhg_field = row.get(mhg_key, "")
            if not enhg:
                continue
            cands = []
            if mhg_field is not None:
                for c in mhg_field.split(";"):
                    c = norm(c)
                    if c:
                        cands.append(c)
            # Only create pairs when an MHG candidate exists
            for cand in cands:
                pairs.append((enhg, cand))
            # If there were no candidates, we still want to register the ENHG node later;
            # we just don't add any cross edges here.
    return pairs


def build_graph(
    enhg_map: Dict[str, str],
    mhg_map: Dict[str, str],
    cross_pairs: Iterable[Tuple[str, str]],
) -> DSU:
    dsu = DSU()

    # Add within-language edges (derived <-> base) for ENHG
    for derived, base in enhg_map.items():
        dsu.union(("ENHG", derived), ("ENHG", base))

    # Add within-language edges for MHG
    for derived, base in mhg_map.items():
        dsu.union(("MHG", derived), ("MHG", base))

    # Ensure presence of all nodes (derived and base) even if isolated
    for lemma in set(list(enhg_map.keys()) + list(enhg_map.values())):
        dsu.add(("ENHG", lemma))
    for lemma in set(list(mhg_map.keys()) + list(mhg_map.values())):
        dsu.add(("MHG", lemma))

    # Add cross-language edges from CSV
    for enhg, mhg in cross_pairs:
        if enhg:
            dsu.add(("ENHG", enhg))
        if mhg:
            dsu.add(("MHG", mhg))
        if enhg and mhg:
            dsu.union(("ENHG", enhg), ("MHG", mhg))

    return dsu


def assign_ids(dsu: DSU) -> Dict[Node, int]:
    """Assign stable, deterministic integer IDs to each connected component."""
    comps: Dict[Node, List[Node]] = defaultdict(list)
    for node in list(dsu.parent.keys()):
        comps[dsu.find(node)].append(node)

    def key_of(node: Node) -> str:
        lang, lemma = node
        return f"{lang}|{lemma}"

    # Sort components by their minimum node key for stability
    ordered_components = sorted(
        (sorted(nodes, key=key_of)[0] for nodes in comps.values()),
        key=key_of,
    )

    id_by_root: Dict[Node, int] = {}
    for idx, rep in enumerate(ordered_components, start=1):
        id_by_root[rep] = idx

    # Map every node to the ID of its component representative (rep is min-key within the comp)
    node_to_id: Dict[Node, int] = {}
    for root, nodes in comps.items():
        # Find canonical representative used in ordered_components
        rep = sorted(nodes, key=key_of)[0]
        cid = id_by_root[rep]
        for n in nodes:
            node_to_id[n] = cid
    return node_to_id


def write_output(
    out_path: Path, node_to_id: Dict[Node, int], sort_by: str = "lemma_id"
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"lemma": lemma, "lemma_id": node_to_id[(lang, lemma)]}
        for (lang, lemma) in node_to_id.keys()
    ]
    # Deterministic sorting for readability
    if sort_by == "lemma_id":
        rows.sort(key=lambda r: (int(r["lemma_id"]), r["lemma"]))
    else:
        rows.sort(key=lambda r: (r["lemma"], int(r["lemma_id"])))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lemma", "lemma_id"])
        writer.writeheader()
        writer.writerows(rows)


def combine_corpus_data(
    enhg_path: Path,
    mhg_path: Path,
    lemma_id_path: Path,
    output_path: Path = Path("data/combined_corpus.csv"),
) -> pd.DataFrame:
    """
    Combine ENHG and MHG corpus data with lemma IDs into a single dataset.

    Args:
        enhg_path: Path to ENHG corpus CSV
        mhg_path: Path to MHG corpus CSV
        lemma_id_path: Path to lemma ID mapping CSV
        output_path: Path to save combined corpus (default: data/combined_corpus.csv)

    Returns:
        pd.DataFrame of combined corpus data
    """
    enhg = pd.read_csv(enhg_path)
    mhg = pd.read_csv(mhg_path)
    lemma_id_map = pd.read_csv(lemma_id_path)

    enhg["corpus"] = "ENHG"
    mhg["corpus"] = "MHG"

    data = pd.concat([mhg, enhg], ignore_index=True)

    data = data.merge(lemma_id_map, on="lemma", how="left")
    data.dropna(subset=["lemma_id"], inplace=True)
    data["lemma_id"] = data["lemma_id"].astype(int)

    if output_path:
        data.to_csv(output_path, index=False)

    return data


def main():
    ap = argparse.ArgumentParser(
        description="Assign shared lemma IDs across ENHG and MHG."
    )
    ap.add_argument(
        "--enhg-mapping",
        type=Path,
        default=Path("data/lemmas/enhg_mapping.json"),
        help="Path to ENHG derived->base mapping JSON",
    )
    ap.add_argument(
        "--mhg-mapping",
        type=Path,
        default=Path("data/lemmas/mhg_mapping.json"),
        help="Path to MHG derived->base mapping JSON",
    )
    ap.add_argument(
        "--cross-csv",
        type=Path,
        default=Path("data/lemmas/etymology_matches_manual.csv"),
        help="Path to ENHG<->MHG base crosslinks CSV",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/lemmas/lemma_id.csv"),
        help="Output CSV path (lemma,lemma_id)",
    )
    ap.add_argument(
        "--enhg_path",
        type=Path,
        default=Path("data/enhg_corpus.csv"),
        help="Path to the ENHG corpus CSV file",
    )
    ap.add_argument(
        "--mhg_path",
        type=Path,
        default=Path("data/mhg_corpus.csv"),
        help="Path to the MHG corpus CSV file",
    )
    ap.add_argument(
        "--sort-by",
        choices=["lemma_id", "lemma"],
        default="lemma_id",
        help="Sort rows by lemma_id (default) or lemma",
    )
    args = ap.parse_args()

    enhg_map = read_mapping(args.enhg_mapping, "ENHG")
    mhg_map = read_mapping(args.mhg_mapping, "MHG")
    cross_pairs = read_crosslinks(args.cross_csv)

    dsu = build_graph(enhg_map, mhg_map, cross_pairs)
    node_to_id = assign_ids(dsu)
    write_output(args.out, node_to_id, sort_by=args.sort_by)

    combined = combine_corpus_data(args.enhg_path, args.mhg_path, args.out)


if __name__ == "__main__":
    main()
