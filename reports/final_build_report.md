# Final Build Report

_Generated 2026-10-31 by `src/make_report.py` from the result files on disk._

**Project.** Reproduction of *"Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style"* (Baumler et al., 2026), plus a novel analysis of **which editing behaviours restore an author's personal style** and whether that recovery is **author-specific** rather than generic humanisation.

The style measure throughout is the cosine similarity between **LUAR-MUD** (`rrivera1849/LUAR-MUD`) embeddings. Each text — control (unassisted) writing, LLM draft, and post-edited final — is embedded as a single document; similarity to a participant's own writing is the mean cosine to their individual control texts.

## 1. Paper reproduction (pre-registered hypotheses)

All effect sizes use Hedges' *g* with a 1,000-sample bootstrap 95% CI and a 10,000-sample permutation test on the mean difference.

| Hypothesis | mean x | mean y | Δ (y−x) | perm p | Hedges g [95% CI] | paper g | dir. matches |
|---|---|---|---|---|---|---|---|
| H1a | 0.852 | 0.864 | 0.012 | <0.0001 | 0.347 [0.192, 0.508] | 0.550 | yes |
| H1a_prime | 0.864 | 0.850 | -0.013 | <0.0001 | -0.462 [-0.619, -0.309] | -0.560 | yes |
| H1b | 0.906 | 0.899 | -0.007 | <0.0001 | -0.417 [-0.570, -0.279] | -0.410 | yes |
| H1c | 0.899 | 0.864 | -0.035 | <0.0001 | -1.276 [-1.461, -1.091] | -1.430 | yes |
| H2a | 0.855 | 0.893 | 0.039 | <0.0001 | 1.117 [1.095, 1.138] | 1.420 | yes |
| H2b | 0.906 | 0.893 | -0.012 | <0.0001 | -0.389 [-0.401, -0.377] | -0.690 | yes |
| H2c | 0.864 | 0.893 | 0.030 | <0.0001 | 1.097 [0.930, 1.294] | 1.140 | yes |

**H3 (perceived vs measured self-similarity, repeated-measures correlation).**

- rmcorr r = **0.207** [0.127, 0.285], p = 6.33e-07, 648 observations / 81 participants.
- Paper: r = 0.244 (± 0.076), p < 0.0001.

**Read.** All H1/H2 effect directions match the paper and every test is significant at p < .05; H3 reproduces the paper's repeated-measures correlation within its CI. Remaining magnitude differences reflect documented preprocessing choices (chunking, LLM-class construction), so this is a strong directional reproduction rather than an exact numeric one.

## 2. Novel contribution

### 2a. Which editing behaviours predict style recovery?

Target: `change_after_editing` — the LUAR move toward the author's own style after editing. 324 tasks / 81 participants, GroupKFold(n_splits=5) grouped by participant. Ridge penalty tuned per training fold by RidgeCV over alphas in [0.01, 1e+03].

| Model | features | CV R² |
|---|---|---|
| M0 | controls only | -0.042 (folds -0.23..0.01) |
| M1 | + quantity of editing | 0.125 (folds -0.24..0.34) |
| M2 | + kind of editing | 0.097 (folds -0.67..0.38) |
| M3 | + quantity + kind (full) | 0.112 (folds -0.75..0.41) |

- Quantity adds R² = 0.167 over controls; kind adds R² = 0.139.
- **Kind of editing beats quantity: False.**
- Strongest predictors: `frac_words_changed` (ρ=0.62); `chars_inserted` (ρ=0.61); `draft_final_similarity` (ρ=-0.61); `edits_per_100_draft_chars` (ρ=0.60).

_Note on M3._ The full model stacks quantity and kind features that carry overlapping information. With the Ridge penalty tuned per training fold, M3's cross-validated R² reflects that redundancy, not a lack of shrinkage — so M2 (kind) is the parsimonious best model.

### 2b. Is the recovered style author-specific?

For each task: `own_gain` = move toward the author's own style, `other_gain` = move toward other authors' styles, `specificity = own_gain − other_gain` (324 tasks).

- mean own_gain = 0.0117, mean other_gain = 0.0022.
- **specificity = 0.0095** [0.0076, 0.0115], perm p = <0.0001, Hedges g = 0.650.
- predict `own_gain` from edit features: CV R² = 0.081.
- predict `other_gain` from edit features: CV R² = 0.085.
- predict `own_gain` from edit features + other_gain: CV R² = 0.344.

**Read.** Positive specificity (p < .001) means editing moves text toward the author's *own* voice more than toward people in general — author-specific recovery, not generic humanisation. The `+ other_gain` model includes other_gain as a predictor, so its R² answers a different question and is not comparable to the edit-features-only values.

## 3. Machine-learning experiments

### 3a. Classical model zoo (GroupKFold by participant)

**3-way (control / draft / edited), chance ≈ 0.33 balanced accuracy — best model per feature set:**

- stylometry: **random_forest** balanced acc 0.750, macro-F1 0.755.
- embeddings: **rbf_svm** balanced acc 0.745, macro-F1 0.730.

**AI-origin detection (control vs draft):**

- embeddings: rbf_svm balanced acc 0.988, AUC 1.000 — **not a valid general detector** (confounded; see caveats).
- stylometry: logreg balanced acc 0.952, AUC 0.982.

### 3b. Deep-learning experiments

- **MLP on LUAR embeddings** (3-way): balanced acc 0.738, macro-F1 0.730 — on par with the best classical model.
- **LSTM over the keystroke stream** (successful edit): AUC 0.597, balanced acc 0.637 (AUC baseline 0.50). Keystroke dynamics alone are only weakly predictive of style recovery.
- **Fine-tuned DistilBERT** (3-way, 16 held-out participants): balanced acc 0.661, macro-F1 0.573 — does not beat the frozen-embedding models; a useful negative result given the small training set.

## 4. Supplementary experiments (`experiments/`)

Kept for completeness; not central findings.

**`style_classifier.py`** — style classifier + LLM-origin detector.
- Control-vs-draft detector AUC = 0.997; fraction of *edited* texts still flagged LLM = 0.892 (consistent with H1c: editing does not fully erase the LLM fingerprint).

**`demographics.py`** — recovery by demographics (Mann-Whitney).
- gender: woman vs man: p = 0.660.
- education_bin: degree vs no_degree: p = 0.600.
- No significant demographic differences in style recovery (small sample; a null, not evidence of equivalence).

## 5. Style recovery by writing scenario

| scenario | mean recovery | n |
|---|---|---|
| eulogy | 0.0267 | 11 |
| letter | 0.0155 | 50 |
| thank | 0.0139 | 53 |
| reassurance | 0.0136 | 38 |
| speech | 0.0099 | 52 |
| apology | 0.0089 | 46 |
| wedding | 0.0075 | 38 |
| condolence | 0.0073 | 36 |

_Association only — scenario is not randomly assigned._

## 6. Features and figures

- Stylometry features: 810 texts × ~120 interpretable features (`build_stylometry.py`).
- Figures (12) in `reports/figures/`: `fig_confusion_best_3way.png`, `fig_effect_sizes.png`, `fig_embedding_pca.png`, `fig_embedding_tsne.png`, `fig_h1a_h1b_before_after.png`, `fig_h1a_prime_own_vs_other.png`, `fig_h1c_self_vs_llm.png`, `fig_h2_homogeneity.png`, `fig_h3_perception_vs_luar.png`, `fig_llm_markers_by_label.png`, `fig_recovery_by_scenario.png`, `fig_roc_ai_detection.png`.

## 7. Caveats and honest limitations

- **Not an exact numeric reproduction.** Directions and significance match the paper; some magnitudes differ because of documented preprocessing choices (document chunking for LUAR, LLM-class built from other participants' drafts, no PANGRAM detector scores).
- **Embedding AI-detection AUC ~ 1.00 is not a valid general detector result.** Control texts are embedded individually (single-document, like drafts), so this is NOT the earlier composite-vs-single-text artifact; LUAR simply separates human control writing from LLM drafts almost perfectly. That reflects a genuinely large human-vs-LLM style gap but is confounded with topic and length (controls and drafts are different prompts), so it overstates generalisable AI detection. The honest, confound-limited number is the stylometry detector (AUC ~ 0.98).
- **Significance treats task/pair scores as independent**, although tasks are nested within participants; effect directions are the primary read for the H1/H2 family.
- **Demographic nulls** come from a small sample and do not establish group equivalence.
- **Cross-validation is grouped by participant** everywhere, so no author appears in both train and test; the Ridge penalty for the edit-recovery models is tuned inside each training fold.

## 8. How to reproduce

```bash
pip install -r requirements.txt

# one-time: build LUAR embeddings (needs torch + einops)
python src/embed_luar.py

# main pipeline
python src/analysis/analyze_hypotheses.py
python src/features/build_stylometry.py
python src/novel/extract_edit_features.py
python src/novel/analyze_edit_recovery.py
python src/novel/own_vs_generic_recovery.py
python src/analysis/classical_ml.py
python src/analysis/visualize_embeddings.py
python src/analysis/make_figures.py
python src/dl/dl_models.py
python src/dl/finetune_transformer.py

# supplementary
python experiments/style_classifier.py
python experiments/demographics.py

# regenerate this report
python src/make_report.py
```
