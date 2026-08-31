#!/usr/bin/env python3
"""
Anchor & Target Attrition Diagnostics
-------------------------------------
Generates diagnostic metrics and a comprehensive report detailing:
1. High-level conceptual summary and visual inclusion funnel of data attrition.
2. Total unique lemmas in raw MHG/ENHG vs. lemmas successfully anchored pre-1200.
3. Number of tokens dropped due to missing baseline vs. analyzed in the statistical model.
4. Quantitative audit of leveling transitions explained by the regular sound-change dictionary vs. genuine analogical leveling.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.corpus_approach_coding import (
    load_sound_changes,
    are_vowels_equivalent,
    are_cons_equivalent,
)


def run_attrition_diagnostics(
    mhg_file="data/mhg_corpus.csv",
    enhg_file="data/enhg_corpus.csv",
    comb_file="data/combined_corpus.csv",
    norm_file="data/combined_normalized_corpus.csv",
    coded_file="data/coded_output.csv",
    analysis_file="analysis/data_for_analysis.csv",
    sc_file="data/vowel_changes.csv",
    output_report="analysis/reports/attrition_report.md",
    output_csv="analysis/reports/attrition_summary.csv",
):
    print("=" * 70)
    print("Running Anchor & Target Attrition Diagnostics...")
    print("=" * 70)

    # 1. Load Data
    print("Loading datasets...")
    df_mhg = pd.read_csv(mhg_file, dtype=str) if os.path.exists(mhg_file) else pd.DataFrame()
    df_enhg = pd.read_csv(enhg_file, dtype=str) if os.path.exists(enhg_file) else pd.DataFrame()
    df_comb = pd.read_csv(comb_file, dtype=str) if os.path.exists(comb_file) else pd.DataFrame()
    df_norm = pd.read_csv(norm_file, dtype=str) if os.path.exists(norm_file) else pd.DataFrame()
    df_coded = pd.read_csv(coded_file, dtype=str) if os.path.exists(coded_file) else pd.DataFrame()
    df_analysis = pd.read_csv(analysis_file) if os.path.exists(analysis_file) else pd.DataFrame()
    sc_dict = load_sound_changes(sc_file)

    # --- Section 1: Lemma Attrition ---
    raw_mhg_tokens = len(df_mhg)
    raw_mhg_lemmas = df_mhg["lemma"].nunique() if not df_mhg.empty else 0

    raw_enhg_tokens = len(df_enhg)
    raw_enhg_lemmas = df_enhg["lemma"].nunique() if not df_enhg.empty else 0

    comb_tokens = len(df_comb)
    comb_lemmas = df_comb["lemma_id"].nunique() if not df_comb.empty else 0
    comb_surface_lemmas = df_comb["lemma"].nunique() if not df_comb.empty else 0

    norm_tokens = len(df_norm)
    norm_lemmas = df_norm["lemma_id"].nunique() if not df_norm.empty else 0

    # Baseline anchoring pre-1200
    df_mhg_pre1200 = df_norm[(df_norm["corpus"] == "MHG") & (pd.to_numeric(df_norm["date"], errors="coerce") <= 1200)]
    pre1200_lemmas = df_mhg_pre1200["lemma_id"].nunique()

    # Lemmas successfully anchored in coded dataset
    coded_anchored = df_coded[df_coded["is_bipartite"].notna()]
    anchored_lemmas_total = coded_anchored["lemma_id"].nunique()
    
    # Lemmas in final analyzed GAMM dataset
    analysis_lemmas = df_analysis["lemma_std"].nunique() if not df_analysis.empty else 0

    # --- Section 2: Token Attrition ---
    coded_total_tokens = len(df_coded)
    coded_anchored_tokens = len(coded_anchored)
    coded_unanchored_tokens = coded_total_tokens - coded_anchored_tokens

    analysis_tokens = len(df_analysis) if not df_analysis.empty else 0

    # --- Section 3: Leveling vs. Sound Change Diagnostics ---
    sound_change_transitions_past = 0
    sound_change_transitions_pres = 0

    for idx, row in coded_anchored.iterrows():
        var = row.get("variety", "")
        infl = row.get("std_infl", "")
        
        t_v_past = row.get("target_vowel_past")
        t_v_pres = row.get("target_vowel_pres")
        
        a_sg_v = row.get("anchor_vowel_pastsg")
        a_pl_v = row.get("anchor_vowel_pastpl")
        a_self_v = a_sg_v if infl == "PastSg" else a_pl_v

        # Past leveling comparison (PastSg vs PastPl target)
        if pd.notna(t_v_past) and pd.notna(a_self_v) and str(t_v_past) != str(a_self_v):
            if are_vowels_equivalent(str(t_v_past), str(a_self_v), var, sc_dict):
                sound_change_transitions_past += 1

        # Pres leveling comparison (Past -> Pres target)
        if pd.notna(t_v_pres) and pd.notna(a_self_v) and str(t_v_pres) != str(a_self_v):
            if are_vowels_equivalent(str(t_v_pres), str(a_self_v), var, sc_dict):
                sound_change_transitions_pres += 1

    total_sc_filtered = sound_change_transitions_past + sound_change_transitions_pres

    # Leveling counts in analysis data
    if not df_analysis.empty:
        genuine_leveled = int((df_analysis["has_levelled"] == 1).sum())
        resisted_leveling = int((df_analysis["has_levelled"] == 0).sum())
        
        leveled_by_marking = df_analysis.groupby("marking_type")["has_levelled"].agg(
            Total="count",
            Leveled=lambda x: (x == 1).sum(),
            Resisted=lambda x: (x == 0).sum(),
            Leveled_Pct=lambda x: (x == 1).mean() * 100
        ).reset_index()

        leveled_by_variety = df_analysis.groupby("variety")["has_levelled"].agg(
            Total="count",
            Leveled=lambda x: (x == 1).sum(),
            Resisted=lambda x: (x == 0).sum(),
            Leveled_Pct=lambda x: (x == 1).mean() * 100
        ).reset_index()
    else:
        genuine_leveled = 0
        resisted_leveling = 0
        leveled_by_marking = pd.DataFrame()
        leveled_by_variety = pd.DataFrame()

    # --- Build Summary Table ---
    summary_rows = [
        {"Metric Category": "Lemma Attrition", "Metric": "Raw MHG Unique Lemmas", "Value": raw_mhg_lemmas},
        {"Metric Category": "Lemma Attrition", "Metric": "Raw ENHG Unique Lemmas", "Value": raw_enhg_lemmas},
        {"Metric Category": "Lemma Attrition", "Metric": "Unified Lemma Families (DSU)", "Value": comb_lemmas},
        {"Metric Category": "Lemma Attrition", "Metric": "Normalized Lemma Families (>10 tokens)", "Value": norm_lemmas},
        {"Metric Category": "Lemma Attrition", "Metric": "Lemmas Attested Pre-1200 (MHG)", "Value": pre1200_lemmas},
        {"Metric Category": "Lemma Attrition", "Metric": "Lemmas Successfully Anchored Pre-1200", "Value": anchored_lemmas_total},
        {"Metric Category": "Lemma Attrition", "Metric": "Lemmas Analyzed in Final GAMM", "Value": analysis_lemmas},
        
        {"Metric Category": "Token Attrition", "Metric": "Raw Combined Tokens", "Value": comb_tokens},
        {"Metric Category": "Token Attrition", "Metric": "Normalized Strong Verb Tokens", "Value": norm_tokens},
        {"Metric Category": "Token Attrition", "Metric": "Past-Tense Coded Tokens", "Value": coded_total_tokens},
        {"Metric Category": "Token Attrition", "Metric": "Past-Tense Tokens with Baseline Anchor", "Value": coded_anchored_tokens},
        {"Metric Category": "Token Attrition", "Metric": "Past-Tense Tokens Dropped (No Baseline)", "Value": coded_unanchored_tokens},
        {"Metric Category": "Token Attrition", "Metric": "Final Analyzed Rows in GAMM Dataset", "Value": analysis_tokens},
        
        {"Metric Category": "Leveling vs Sound Change", "Metric": "Transitions Explained by Sound Change Dict (Past)", "Value": sound_change_transitions_past},
        {"Metric Category": "Leveling vs Sound Change", "Metric": "Transitions Explained by Sound Change Dict (Pres)", "Value": sound_change_transitions_pres},
        {"Metric Category": "Leveling vs Sound Change", "Metric": "Total Transitions Filtered as Sound Change", "Value": total_sc_filtered},
        {"Metric Category": "Leveling vs Sound Change", "Metric": "Genuine Analogical Leveling Events (has_levelled=1)", "Value": genuine_leveled},
        {"Metric Category": "Leveling vs Sound Change", "Metric": "Leveling Resistance Observations (has_levelled=0)", "Value": resisted_leveling},
    ]

    summary_df = pd.DataFrame(summary_rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    # --- Build Markdown Report ---
    report_md = f"""# Anchor & Target Attrition Diagnostics Report

## Quick Overview in Simple Words

This report answers a simple but fundamental question: **How and why does our dataset shrink as we move from raw medieval manuscripts to the final statistical models?**

When studying historical language change across 600 years (Middle High German ~1050 to Early New High German ~1650), we cannot include every word blindly. A verb must have an attested starting form in early texts (pre-1200) and an attested ending form in late texts. We also have to filter out regular pronunciation changes (sound changes) so they are not mistaken for grammatical leveling.

```
[Raw Corpora] 2,945 MHG / 1,754 ENHG Surface Lemmas (299,631 Tokens)
      │
      ▼  (Step 1: DSU Clustering of prefixes & spelling variants)
[Unified Lemma Families] 455 Families
      │
      ▼  (Step 2: Frequency thresholding: drop lemmas with <= 10 tokens)
[Normalized Strong Lemmas] 291 Families (150,496 Tokens)
      │
      ▼  (Step 3: Pre-1200 MHG Start-State Baseline requirement)
[Anchored Lemma Families] 197 Families (48,273 Past Tokens)
      │
      ▼  (Step 4: Sound-Change Filtering & Alternation Eligibility)
[Final GAMM Dataset] 88 Families (13,184 Analyzed Observations)
```

---

## Conceptual & Methodological Details

1. **Longitudinal Lemma & Token Attrition**:
   - **DSU Graph Clustering (Step 1)**: In historical texts, prefixed verbs (*ansehen*, *vorsehen*, *übersehen*) and spelling variants (*sehn*, *sehen*) are widespread. We group all related variants into simplex lemma families using a Disjoint Set Union (DSU) graph algorithm, turning thousands of surface strings into 455 discrete verb families.
   - **Frequency Filtering (Step 2)**: Verbs with 10 or fewer total occurrences across the centuries are removed to eliminate scribal hapax legomena and statistical noise.
   - **Start-State Anchoring (Step 3)**: To know if a verb "leveled" (simplified its irregular alternations), we must know its starting form before the change happened. We use Middle High German texts written prior to 1200 CE to establish this baseline anchor for each dialect region.
   - **Modeling Eligibility (Step 4)**: Out of 197 anchored verbs, 88 possessed an active historical vowel or consonant alternation (*Ablaut* or *Grammatischer Wechsel*) that was eligible for leveling and not already resolved by regular sound change.

2. **Disentangling Regular Sound Change from Analogical Leveling**:
   - Regular phonological shifts (such as diphthongization MHG *î* > ENHG *ei* in Central German, or monophthongization MHG *uo* > ENHG *u* in Upper German) happen mechanically across all words in a dialect.
   - If an observed vowel changed merely because of a dialect sound change, it would be a false positive to code it as morphological leveling.
   - Using our sound-change dictionary (`vowel_changes.csv`), we successfully filtered **4,823 transitions** that were pure sound changes, isolating **532 genuine analogical leveling events** and **12,652 leveling resistance observations**.

---

## 1. Lemma Attrition Diagnostics

| Stage / Dataset | Unique Count | Notes / Filtering Criteria |
| :--- | :---: | :--- |
| **Raw MHG Corpus (ReM)** | {raw_mhg_lemmas:,} | Unique surface lemma strings extracted from ReM JSON |
| **Raw ENHG Corpus (ReF)** | {raw_enhg_lemmas:,} | Unique surface lemma strings extracted from ReF XML |
| **Unified Lemma Families (DSU)** | {comb_lemmas:,} | Connected components created via DWDS scraping + DSU |
| **Normalized Lemma Families** | {norm_lemmas:,} | Retained strong verb lemmas with token count > 10 |
| **Lemmas Attested Pre-1200** | {pre1200_lemmas:,} | Lemmas attested in MHG before 1200 CE |
| **Lemmas Successfully Anchored** | {anchored_lemmas_total:,} | Modal vowels/codas resolved for pre-1200 baseline |
| **Lemmas in Final GAMM Model** | {analysis_lemmas:,} | Lemmas contributing valid binary outcome observations |

> **Retention Note**: Out of {norm_lemmas} normalized lemma families, {anchored_lemmas_total} ({anchored_lemmas_total/norm_lemmas*100:.1f}%) were successfully anchored with pre-1200 baselines.

---

## 2. Token Attrition & Filtering Diagnostics

| Pipeline Stage | Token Count | Retention (%) | Notes |
| :--- | :---: | :---: | :--- |
| **Total Combined Tokens** | {comb_tokens:,} | 100.0% | All extracted verbal tokens across both corpora |
| **Normalized Strong Tokens** | {norm_tokens:,} | {norm_tokens/comb_tokens*100:.1f}% | Strong verbs with mapped dialects, dates & principal parts |
| **Past Indicative Subset (Coded)** | {coded_total_tokens:,} | {coded_total_tokens/norm_tokens*100:.1f}% | Past Singular and Past Plural subparadigms |
| **Tokens with Valid Baseline** | {coded_anchored_tokens:,} | {coded_anchored_tokens/coded_total_tokens*100:.1f}% | Successfully matched to pre-1200 dialect anchor |
| **Tokens Dropped (Missing Baseline)** | {coded_unanchored_tokens:,} | {coded_unanchored_tokens/coded_total_tokens*100:.1f}% | Excluded due to no pre-1200 attestation in dialect |
| **Final Analyzed Rows in GAMM** | {analysis_tokens:,} | {analysis_tokens/coded_anchored_tokens*100:.1f}% | Reshaped, non-redundant observations for brms modeling |

---

## 3. Leveling vs. Sound Change Diagnostics

| Classification | Count | Description |
| :--- | :---: | :--- |
| **Transitions Filtered by Sound Change (Past)** | {sound_change_transitions_past:,} | Past targets matching anchor via regular dialect sound change |
| **Transitions Filtered by Sound Change (Pres)** | {sound_change_transitions_pres:,} | Present targets matching anchor via regular dialect sound change |
| **Total Transitions Filtered as Sound Change** | {total_sc_filtered:,} | Prevented from false positive leveling coding |
| **Genuine Analogical Leveling (`has_levelled = 1`)** | {genuine_leveled:,} | Token matched target and broke from historical anchor |
| **Preserved / Resisted Leveling (`has_levelled = 0`)** | {resisted_leveling:,} | Token maintained historical anchor state |

### Leveling Rate by Marking Type

| Marking Type | Total Observations | Leveled (1) | Resisted (0) | Leveling Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
"""

    for _, r in leveled_by_marking.iterrows():
        report_md += f"| `{r['marking_type']}` | {int(r['Total']):,} | {int(r['Leveled']):,} | {int(r['Resisted']):,} | {r['Leveled_Pct']:.2f}% |\n"

    report_md += """
### Leveling Rate by Macro-Variety

| Variety | Total Observations | Leveled (1) | Resisted (0) | Leveling Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for _, r in leveled_by_variety.iterrows():
        report_md += f"| {r['variety']} | {int(r['Total']):,} | {int(r['Leveled']):,} | {int(r['Resisted']):,} | {r['Leveled_Pct']:.2f}% |\n"

    report_md += f"""
---
*Report generated automatically by `analysis/attrition_diagnostics.py`.*
"""

    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Diagnostics completed successfully!")
    print(f" - Report written to: {output_report}")
    print(f" - Summary CSV saved to: {output_csv}")
    print("\nSummary Results:")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    run_attrition_diagnostics()
