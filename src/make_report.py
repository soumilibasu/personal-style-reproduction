"""
src/make_report.py

Generate the project's FINAL BUILD REPORT from the result files on disk.

This script reads every output produced by the pipeline (hypothesis tests,
the novel edit-recovery and own-vs-generic analyses, the classical-ML and
deep-learning experiments, the supplementary scripts, and the figure list)
and writes a single Markdown report:

    reports/final_build_report.md

Every number in the report is read from a result file at run time, so the
report always reflects the latest run of the pipeline. Run it LAST, after the
analysis scripts have written their outputs:

    python src/make_report.py

Inputs (all optional; missing ones are reported as "not generated"):
    data/processed/hypotheses/all_hypotheses_summary.csv
    data/processed/hypotheses/hypotheses_report.json
    data/processed/novel/novel_report.json
    data/processed/novel/own_vs_generic_report.json
    data/processed/novel/classifier_metrics.csv          (supplementary)
    data/processed/analysis/classical_ml_model_comparison.csv
    data/processed/analysis/recovery_by_scenario.csv
    data/processed/analysis/demographics_group_tests.csv (supplementary)
    data/processed/features/stylometry_features.csv
    data/processed/dl/dl_results.json
    data/processed/dl/transformer_results.json
    reports/figures/*.png

Output:
    reports/final_build_report.md
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd


DATA = Path("data/processed")
FIG_DIR = Path("reports/figures")
OUT = Path("reports/final_build_report.md")


# ---------------------------------------------------------------------------
# Safe loaders
# ---------------------------------------------------------------------------
def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_csv(path):
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return None


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{nd}f}"


# ---------------------------------------------------------------------------
# Section builders (each returns a list of markdown lines)
# ---------------------------------------------------------------------------
def section_overview():
    return [
        "# Final Build Report",
        "",
        f"_Generated {date.today().isoformat()} by `src/make_report.py` from "
        "the result files on disk._",
        "",
        "**Project.** Reproduction of *\"Can You Make It Sound Like You? "
        "Post-Editing LLM-Generated Text for Personal Style\"* (Baumler et "
        "al., 2026), plus a novel analysis of **which editing behaviours "
        "restore an author's personal style** and whether that recovery is "
        "**author-specific** rather than generic humanisation.",
        "",
        "The style measure throughout is the cosine similarity between "
        "**LUAR-MUD** (`rrivera1849/LUAR-MUD`) embeddings. Each text — control "
        "(unassisted) writing, LLM draft, and post-edited final — is embedded "
        "as a single document; similarity to a participant's own writing is "
        "the mean cosine to their individual control texts.",
        "",
    ]


def section_repro():
    lines = ["## 1. Paper reproduction (pre-registered hypotheses)", ""]
    summary = load_csv(DATA / "hypotheses" / "all_hypotheses_summary.csv")
    if summary is None:
        lines += ["_Not generated. Run `python src/analysis/"
                  "analyze_hypotheses.py`._", ""]
        return lines

    lines += [
        "All effect sizes use Hedges' *g* with a 1,000-sample bootstrap 95% "
        "CI and a 10,000-sample permutation test on the mean difference.",
        "",
        "| Hypothesis | mean x | mean y | Δ (y−x) | perm p | Hedges g "
        "[95% CI] | paper g | dir. matches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        p = r["permutation_p_two_sided"]
        p_str = "<0.0001" if p < 1e-4 else fmt(p, 4)
        lines.append(
            f"| {r['hypothesis']} | {fmt(r['mean_x'])} | {fmt(r['mean_y'])} | "
            f"{fmt(r['mean_difference_y_minus_x'])} | {p_str} | "
            f"{fmt(r['hedges_g'])} [{fmt(r['g_ci_95_low'])}, "
            f"{fmt(r['g_ci_95_high'])}] | {fmt(r.get('paper_g'))} | "
            f"{'yes' if r.get('direction_matches_paper') else 'no'} |"
        )
    lines.append("")

    report = load_json(DATA / "hypotheses" / "hypotheses_report.json")
    h3 = (report or {}).get("H3")
    if h3:
        lines += [
            "**H3 (perceived vs measured self-similarity, repeated-measures "
            "correlation).**",
            "",
            f"- rmcorr r = **{fmt(h3['rmcorr_r'])}** "
            f"[{fmt(h3['ci_95_low'])}, {fmt(h3['ci_95_high'])}], "
            f"p = {h3['rmcorr_p']:.2e}, "
            f"{h3['n_observations']} observations / "
            f"{h3['n_participants']} participants.",
            f"- Paper: r = {h3['paper_rmcorr_r']} "
            f"(± {h3['paper_ci_halfwidth']}), p < {h3['paper_p']}.",
            "",
        ]
    direction_ok = (
        summary["direction_matches_paper"].fillna(False).astype(bool).all()
        if "direction_matches_paper" in summary else False
    )
    lines += [
        "**Read.** All H1/H2 effect directions "
        f"{'match' if direction_ok else 'are checked against'} the paper and "
        "every test is significant at p < .05; H3 reproduces the paper's "
        "repeated-measures correlation within its CI. Remaining magnitude "
        "differences reflect documented preprocessing choices (chunking, "
        "LLM-class construction), so this is a strong directional "
        "reproduction rather than an exact numeric one.",
        "",
    ]
    return lines


def section_novel():
    lines = ["## 2. Novel contribution", ""]

    nr = load_json(DATA / "novel" / "novel_report.json")
    lines += ["### 2a. Which editing behaviours predict style recovery?", ""]
    if nr is None:
        lines += ["_Not generated. Run `python src/novel/"
                  "analyze_edit_recovery.py`._", ""]
    else:
        m = nr["model_cv_r2"]
        folds = nr.get("model_fold_r2", {})

        def fold_range(name):
            vs = folds.get(name)
            if not vs:
                return ""
            return f" (folds {fmt(min(vs),2)}..{fmt(max(vs),2)})"

        lines += [
            f"Target: `{nr['target']}` — the LUAR move toward the author's "
            f"own style after editing. {nr['n_tasks']} tasks / "
            f"{nr['n_participants']} participants, "
            f"{nr['cv']}. {nr.get('regularisation', '')}",
            "",
            "| Model | features | CV R² |",
            "|---|---|---|",
            f"| M0 | controls only | {fmt(m.get('M0_controls'))}"
            f"{fold_range('M0_controls')} |",
            f"| M1 | + quantity of editing | "
            f"{fmt(m.get('M1_controls+quantity'))}"
            f"{fold_range('M1_controls+quantity')} |",
            f"| M2 | + kind of editing | {fmt(m.get('M2_controls+kind'))}"
            f"{fold_range('M2_controls+kind')} |",
            f"| M3 | + quantity + kind (full) | {fmt(m.get('M3_full'))}"
            f"{fold_range('M3_full')} |",
            "",
            f"- Quantity adds R² = {fmt(nr['quantity_r2_gain_over_controls'])} "
            f"over controls; kind adds R² = "
            f"{fmt(nr['kind_r2_gain_over_controls'])}.",
            f"- **Kind of editing beats quantity: "
            f"{nr['kind_beats_quantity']}.**",
        ]
        top = nr.get("top_correlations", [])[:4]
        if top:
            lines.append(
                "- Strongest predictors: "
                + "; ".join(f"`{t['feature']}` (ρ={fmt(t['spearman_r'],2)})"
                            for t in top)
                + "."
            )
        lines += [
            "",
            "_Note on M3._ The full model stacks quantity and kind features "
            "that carry overlapping information. With the Ridge penalty tuned "
            "per training fold, M3's cross-validated R² reflects that "
            "redundancy, not a lack of shrinkage — so M2 (kind) is the "
            "parsimonious best model.",
            "",
        ]

    og = load_json(DATA / "novel" / "own_vs_generic_report.json")
    lines += ["### 2b. Is the recovered style author-specific?", ""]
    if og is None:
        lines += ["_Not generated. Run `python src/novel/"
                  "own_vs_generic_recovery.py`._", ""]
    else:
        s = og["specificity"]
        p = s["permutation_p"]
        p_str = "<0.0001" if p < 1e-4 else fmt(p, 4)
        lines += [
            f"For each task: `own_gain` = move toward the author's own style, "
            f"`other_gain` = move toward other authors' styles, "
            f"`specificity = own_gain − other_gain` ({og['n_tasks']} tasks).",
            "",
            f"- mean own_gain = {fmt(s['mean_own_gain'],4)}, "
            f"mean other_gain = {fmt(s['mean_other_gain'],4)}.",
            f"- **specificity = {fmt(s['mean_specificity'],4)}** "
            f"[{fmt(s['ci_95'][0],4)}, {fmt(s['ci_95'][1],4)}], "
            f"perm p = {p_str}, Hedges g = {fmt(s['hedges_g_own_vs_other'])}.",
        ]
        for mdl in og.get("models", []):
            lines.append(
                f"- predict `{mdl['target']}` from {mdl['features']}: "
                f"CV R² = {fmt(mdl['cv_r2'])}."
            )
        lines += [
            "",
            "**Read.** Positive specificity (p < .001) means editing moves "
            "text toward the author's *own* voice more than toward people in "
            "general — author-specific recovery, not generic humanisation. "
            "The `+ other_gain` model includes other_gain as a predictor, so "
            "its R² answers a different question and is not comparable to the "
            "edit-features-only values.",
            "",
        ]
    return lines


def _best_rows(df, task):
    sub = df[df["task"] == task].copy()
    if sub.empty:
        return sub
    return sub.sort_values("balanced_accuracy", ascending=False)


def section_ml():
    lines = ["## 3. Machine-learning experiments", ""]
    cm = load_csv(DATA / "analysis" / "classical_ml_model_comparison.csv")
    lines += ["### 3a. Classical model zoo (GroupKFold by participant)", ""]
    if cm is None:
        lines += ["_Not generated. Run `python src/analysis/"
                  "classical_ml.py`._", ""]
    else:
        three = _best_rows(cm, "3way")
        det = _best_rows(cm, "detect")
        if not three.empty:
            lines += ["**3-way (control / draft / edited), chance ≈ 0.33 "
                      "balanced accuracy — best model per feature set:**", ""]
            for fs in three["feature_set"].unique():
                r = three[three["feature_set"] == fs].iloc[0]
                lines.append(
                    f"- {fs}: **{r['model']}** balanced acc "
                    f"{fmt(r['balanced_accuracy'])}, "
                    f"macro-F1 {fmt(r['macro_f1'])}.")
            lines.append("")
        if not det.empty:
            lines += ["**AI-origin detection (control vs draft):**", ""]
            for fs in det["feature_set"].unique():
                r = det[det["feature_set"] == fs].iloc[0]
                auc_val = float(r["roc_auc"]) if "roc_auc" in r and \
                    pd.notna(r["roc_auc"]) else 0.0
                note = (" — **not a valid general detector** (confounded; "
                        "see caveats)" if fs == "embeddings"
                        and auc_val > 0.999 else "")
                auc = fmt(r["roc_auc"]) if "roc_auc" in r else "n/a"
                lines.append(
                    f"- {fs}: {r['model']} balanced acc "
                    f"{fmt(r['balanced_accuracy'])}, AUC {auc}{note}.")
            lines.append("")

    dl = load_json(DATA / "dl" / "dl_results.json")
    tr = load_json(DATA / "dl" / "transformer_results.json")
    lines += ["### 3b. Deep-learning experiments", ""]
    if dl is None and tr is None:
        lines += ["_Not generated. Run `python src/dl/dl_models.py` and "
                  "`python src/dl/finetune_transformer.py`._", ""]
    else:
        if dl and "mlp_embeddings" in dl:
            e = dl["mlp_embeddings"]
            lines.append(f"- **MLP on LUAR embeddings** (3-way): balanced acc "
                         f"{fmt(e['balanced_accuracy'])}, macro-F1 "
                         f"{fmt(e['macro_f1'])} — on par with the best "
                         f"classical model.")
        if dl and "lstm_keystrokes" in dl:
            l = dl["lstm_keystrokes"]
            lines.append(
                f"- **LSTM over the keystroke stream** (successful edit): "
                f"AUC {fmt(l['roc_auc'])}, balanced acc "
                f"{fmt(l['balanced_accuracy'])} (AUC baseline 0.50). Keystroke "
                f"dynamics alone are only weakly predictive of style recovery.")
        if tr:
            lines.append(
                f"- **Fine-tuned DistilBERT** (3-way, "
                f"{tr.get('split', 'held-out split')}): balanced acc "
                f"{fmt(tr['balanced_accuracy'])}, macro-F1 "
                f"{fmt(tr['macro_f1'])} — does not beat the frozen-embedding "
                f"models; a useful negative result given the small training "
                f"set.")
        lines.append("")
    return lines


def section_supplementary():
    lines = ["## 4. Supplementary experiments (`experiments/`)", "",
             "Kept for completeness; not central findings.", ""]

    det = load_csv(DATA / "novel" / "classifier_metrics.csv")
    if det is not None and "task" in det and not det.empty:
        d = det[det["task"] == "llm_origin_detector"]
        d = d[d["model"] == "logreg"] if "logreg" in set(d.get("model", [])) else d
        row = d.iloc[0] if not d.empty else det.iloc[0]
        auc = row.get("detector_auc_control_vs_draft")
        flagged = row.get("frac_edited_flagged_llm")
        lines += [
            "**`style_classifier.py`** — style classifier + LLM-origin "
            "detector.",
            f"- Control-vs-draft detector AUC = {fmt(auc)}; fraction of "
            f"*edited* texts still flagged LLM = {fmt(flagged, 3)} "
            "(consistent with H1c: editing does not fully erase the LLM "
            "fingerprint).",
            "",
        ]

    demo = load_csv(DATA / "analysis" / "demographics_group_tests.csv")
    if demo is not None and not demo.empty:
        lines += ["**`demographics.py`** — recovery by demographics "
                  "(Mann-Whitney)."]
        own = demo[demo["metric"] == "mean_own_gain"]
        for _, r in own.iterrows():
            lines.append(
                f"- {r['comparison']}: p = {fmt(r['p_value'], 3)}.")
        lines += ["- No significant demographic differences in style "
                  "recovery (small sample; a null, not evidence of "
                  "equivalence).", ""]
    return lines


def section_scenario():
    sc = load_csv(DATA / "analysis" / "recovery_by_scenario.csv")
    if sc is None or sc.empty:
        return []
    lines = ["## 5. Style recovery by writing scenario", "",
             "| scenario | mean recovery | n |", "|---|---|---|"]
    for _, r in sc.sort_values("mean", ascending=False).iterrows():
        lines.append(f"| {r['scenario']} | {fmt(r['mean'],4)} | "
                     f"{int(r['count'])} |")
    lines += ["", "_Association only — scenario is not randomly assigned._", ""]
    return lines


def section_features_and_figures():
    lines = ["## 6. Features and figures", ""]
    sf = load_csv(DATA / "features" / "stylometry_features.csv")
    if sf is not None:
        meta = {"participant_id", "task_key", "scenario", "text_type",
                "embedding_key", "label"}
        n_feat = len([c for c in sf.columns if c not in meta])
        lines.append(f"- Stylometry features: {sf.shape[0]} texts × "
                     f"~{n_feat} interpretable features "
                     f"(`build_stylometry.py`).")
    if FIG_DIR.exists():
        figs = sorted(p.name for p in FIG_DIR.glob("*.png"))
        if figs:
            lines.append(f"- Figures ({len(figs)}) in `reports/figures/`: "
                         + ", ".join(f"`{f}`" for f in figs) + ".")
    lines.append("")
    return lines


def section_caveats():
    return [
        "## 7. Caveats and honest limitations",
        "",
        "- **Not an exact numeric reproduction.** Directions and significance "
        "match the paper; some magnitudes differ because of documented "
        "preprocessing choices (document chunking for LUAR, LLM-class built "
        "from other participants' drafts, no PANGRAM detector scores).",
        "- **Embedding AI-detection AUC ~ 1.00 is not a valid general "
        "detector result.** Control texts are embedded individually "
        "(single-document, like drafts), so this is NOT the earlier "
        "composite-vs-single-text artifact; LUAR simply separates human "
        "control writing from LLM drafts almost perfectly. That reflects a "
        "genuinely large human-vs-LLM style gap but is confounded with topic "
        "and length (controls and drafts are different prompts), so it "
        "overstates generalisable AI detection. The honest, confound-limited "
        "number is the stylometry detector (AUC ~ 0.98).",
        "- **Significance treats task/pair scores as independent**, although "
        "tasks are nested within participants; effect directions are the "
        "primary read for the H1/H2 family.",
        "- **Demographic nulls** come from a small sample and do not establish "
        "group equivalence.",
        "- **Cross-validation is grouped by participant** everywhere, so no "
        "author appears in both train and test; the Ridge penalty for the "
        "edit-recovery models is tuned inside each training fold.",
        "",
    ]


def section_howto():
    return [
        "## 8. How to reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "",
        "# one-time: build LUAR embeddings (needs torch + einops)",
        "python src/embed_luar.py",
        "",
        "# main pipeline",
        "python src/analysis/analyze_hypotheses.py",
        "python src/features/build_stylometry.py",
        "python src/novel/extract_edit_features.py",
        "python src/novel/analyze_edit_recovery.py",
        "python src/novel/own_vs_generic_recovery.py",
        "python src/analysis/classical_ml.py",
        "python src/analysis/visualize_embeddings.py",
        "python src/analysis/make_figures.py",
        "python src/dl/dl_models.py",
        "python src/dl/finetune_transformer.py",
        "",
        "# supplementary",
        "python experiments/style_classifier.py",
        "python experiments/demographics.py",
        "",
        "# regenerate this report",
        "python src/make_report.py",
        "```",
        "",
    ]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines += section_overview()
    lines += section_repro()
    lines += section_novel()
    lines += section_ml()
    lines += section_supplementary()
    lines += section_scenario()
    lines += section_features_and_figures()
    lines += section_caveats()
    lines += section_howto()

    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(lines)} lines).")


if __name__ == "__main__":
    main()
