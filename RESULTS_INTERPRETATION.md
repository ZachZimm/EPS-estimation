# Results Interpretation

## Table of Contents

- Executive Summary
- Objective
- Data and Evaluation Setup
- Modeling Approaches Tried
- High-Level Results
- What Worked Best
- Main Conclusions
- How to Interpret the Learnable-Space Analysis
- Practical Takeaway

## Executive Summary

This repo explored whether next-quarter announced EPS can be predicted from information available before the announcement date. The main result is that there is real signal, but the task is harder and more tail-dominated than it may first appear.

The strongest learned model in the experiments was the trimmed expanded global LSTM point model:
- test MAE `0.8352`
- test RMSE `1.4085`

That is a meaningful result, but it still did not beat the strongest simple baseline on the same trimmed universe:
- trailing-mean baseline MAE `0.7797`

The most important conclusions are:
- predicting residual change from last EPS worked better than predicting raw EPS
- larger training universes helped
- volatility trimming helped a lot by removing catastrophic tail names
- LSTM outperformed the transformer on the trimmed universe
- sector-specific modeling and added fundamentals were useful infrastructure but did not become the new best system
- neural quantile models were overconfident, while true linear quantile regression was a more credible uncertainty baseline

The learnable-space analysis supports the same story. Most of the problem is learnable near flat or modest EPS changes, while large positive or negative EPS jumps are much harder and dominate error tails.

## Objective

The goal of this project is to predict a company’s next announced quarterly `BasicEPS` using only information that would have been available before the EPS announcement date.

This is an event-level forecasting problem, not a daily prediction problem. Each supervised sample corresponds to one EPS announcement event. The target is the announced quarterly EPS, and the inputs are restricted to information that was public strictly before that event’s inferred `publishedDate`.

Two practical questions drove the experiments:

1. Is there enough signal in price history, prior EPS, macro context, and public fundamentals to beat simple historical baselines?
2. What parts of the target space appear learnable versus structurally unstable?

## Data and Evaluation Setup

### Dataset construction

The dataset builder creates one sample per quarterly EPS announcement.

For each sample:
- the target is the next announced quarterly `BasicEPS`
- the event date is the inferred `publishedDate`, not `asOfDate`
- the sequence input is a fixed lookback window of pre-announcement market data
- the static input includes prior EPS history, timing features, and later experiments added pre-release fundamentals and macro context

The published-date alignment logic is intentionally conservative. If a filing date is not directly available, the system estimates when the data became public and only allows features from before that point.

### Splitting and leakage control

Train/validation/test are time-based splits using `target_published_date`:
- train: oldest period
- validation: middle period
- test: most recent period

This is the correct evaluation setup for deployment-style forecasting. A random split would have produced more optimistic but less meaningful results.

### Baselines

Several simple baselines were evaluated. In practice, the strongest baseline was usually:
- trailing mean of prior EPS, defined here as the mean of the last four published quarterly EPS values

Other baselines included:
- persistence: last published quarterly EPS
- seasonal naive: same fiscal quarter one year earlier
- trend baseline
- sector peer median

A recurring result throughout the experiments is that the learned models often beat persistence, but much more rarely beat the trailing-mean baseline.

## Modeling Approaches Tried

The main experimental families were:

- Global transformer models
- Per-sector transformer models
- Macro-augmented models
- Pre-release fundamentals and sector-specific modeling
- Volatility-trimmed universes
- LSTM replacements for the transformer
- Quantile versions of the neural models
- A true linear baseline family

The most important target formulation change was:
- predict `delta_last`, meaning next EPS minus last known EPS

That residual target worked materially better than trying to predict raw EPS directly.

## High-Level Results

## 1. Early transformer work showed signal, but not a strong universal forecaster

A strong early full-run transformer produced:
- test MAE `1.0758`
- test RMSE `3.7838`

That was enough to show real signal, but not enough to be compelling on its own. Tail names dominated the error profile.

## 2. Macro features helped only after the feature engineering was fixed

The first macro version dropped too many samples because of warmup requirements. After reworking the macro feature engineering to preserve more usable rows, macro data became mildly helpful.

The best 50-ticker macro v2 transformer produced:
- test MAE `1.0743`
- test RMSE `3.7063`

This was only a small improvement over the earlier non-macro full run, but it established that macro context was directionally useful when engineered carefully.

## 3. Expanding the universe helped more than expected

The best broad global macro transformer run on the expanded universe produced:
- test MAE `0.9517`
- test RMSE `2.8733`

This was a major improvement over the narrower 50-ticker runs.

Interpretation:
- more cross-ticker data helped generalization
- broader coverage improved the global model more than expected
- the problem did not appear to be “too many tickers” by itself

## 4. Sector-specific modeling and pre-release fundamentals were mixed

The full phase-two sector run, which added pre-release fundamentals and trained one model per sector bucket, produced:
- test MAE `0.9988`
- test RMSE `2.9386`

This beat the older narrower runs, but it did not beat the best expanded global macro transformer.

More importantly, it did not beat the strongest simple baseline on that same test set:
- model MAE `0.9988`
- trailing-mean baseline MAE `0.9363`

Interpretation:
- the additional infrastructure was useful
- the added complexity was not yet justified by the performance gain
- sector splitting did not solve the heavy-tail problem

### Per-sector results

The sector runs were not uniformly bad. They were uneven. Some sectors were modeled reasonably well, while others remained structurally difficult.

Representative sector MAE results from the full phase-two sector run:
- Consumer Defensive: `0.3543`
- Utilities: `0.5717`
- Financial Services: `0.7747`
- Technology: `0.7805`
- Communication Services: `0.8990`
- Industrials: `1.0530`
- Healthcare: `1.2179`
- Consumer Cyclical: `2.1425`

Interpretation:
- stable sectors such as Consumer Defensive and Utilities were much easier
- Technology and Financial Services were workable but not dominant wins
- Consumer Cyclical remained the clearest weak sector and was heavily affected by tail names
- after volatility trimming, sector performance improved substantially, but the trimmed global model still slightly outperformed the trimmed per-sector model overall

## 5. Volatility trimming was one of the clearest improvements

Removing the top 10% most volatile price-history tickers helped substantially.

The corrected trimmed expanded global transformer produced:
- test MAE `0.8660`
- test RMSE `1.4447`

This was a large improvement over the untrimmed expanded global transformer.

Interpretation:
- a small set of very high-volatility names contributed disproportionately to error tails
- trimming makes the task materially easier
- most of the gain comes from removing catastrophic failures rather than improving ordinary cases

Even after trimming, the strongest simple baseline on that trimmed universe was still better:
- trailing-mean baseline MAE `0.7797`

## 6. LSTM was better than the transformer on the trimmed universe

The best learned point model in the architecture sweep was the trimmed expanded global LSTM:
- test MAE `0.8352`
- test RMSE `1.4085`

Compared with the trimmed expanded global transformer:
- transformer: `0.8660` / `1.4447`
- LSTM: `0.8352` / `1.4085`

Interpretation:
- the transformer was not the universally best sequence model here
- once the worst tail names were removed, the simpler recurrent model generalized slightly better

This is the strongest learned point-model result in the repo so far.

It still did not beat the trailing-mean baseline on the trimmed set:
- LSTM MAE `0.8352`
- trailing mean `0.7797`

## 7. Quantile models added interval outputs, but the neural ones were overconfident

Quantile versions of the neural models generally produced intervals with poor empirical coverage.

Example, trimmed expanded global transformer quantile run:
- test MAE `0.8328`
- interval 80% coverage `0.3646`

Trimmed expanded global LSTM quantile run:
- test MAE `0.8338`
- interval 80% coverage `0.3849`

Interpretation:
- the median forecast was sometimes competitive
- the raw interval estimates were too narrow
- the uncertainty estimates were not trustworthy as calibrated distributions

This is why conformal calibration was flagged as a worthwhile future direction.

## 8. True linear baseline results split into two very different stories

### True linear point regression

The true linear point runs were poor.

Examples:
- expanded global true linear point: MAE `3.3252`
- trimmed global true linear point: MAE `4.1044`

These are far worse than the learned neural models and worse than the simple trailing-mean baseline.

Interpretation:
- plain OLS is too brittle here
- a linear point model is not a competitive forecaster for this task

### True linear quantile regression

The true linear quantile runs were much more useful.

Best true linear result:
- trimmed expanded global true linear quantile: MAE `0.8180`
- interval 80% coverage `0.5757`

This is still worse than the trailing-mean baseline on the same set:
- trailing mean `0.7797`

But it is interesting because:
- it is materially better than the true linear point model
- it is competitive with the neural quantile models on MAE
- its interval coverage is better than the neural quantile models, though still below 80%

Interpretation:
- linear quantile regression is a credible baseline family
- plain linear point regression is not

## What Worked Best

If “best” means best learned point model:
- trimmed expanded global LSTM point model
- test MAE `0.8352`
- test RMSE `1.4085`

If “best” means best simple baseline on the trimmed global universe:
- trailing mean baseline
- test MAE `0.7797`

This distinction matters. The strongest learned model in the repo still does not beat the strongest simple baseline consistently.

## Main Conclusions

### 1. There is real signal

The experiments are not a total failure. The models do learn nontrivial structure and often beat persistence.

### 2. The task is dominated by a hard tail

A relatively small subset of names contributes a disproportionate amount of error. Volatility trimming made this very obvious.

### 3. Residual prediction was the right target reformulation

Predicting next EPS directly was weaker than predicting the change relative to last known EPS.

### 4. Broad data helped, but complexity did not always pay off

Adding more tickers helped more than adding complicated sector-specific training.

### 5. The strongest benchmark is not persistence

The right benchmark is usually the trailing-mean baseline, not last-quarter persistence.

### 6. Raw uncertainty estimates remain a weak point

Neural quantile models were badly under-covered. True linear quantile regression did better, but still not enough to be considered calibrated.

## How to Interpret the Learnable-Space Analysis

The learnable-space outputs are in [artifacts/learnable_space_analysis/](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis).

These plots are not model explanations in the SHAP sense. They are geometric and response-surface diagnostics designed to answer a different question:
- where in feature-target space does this problem appear structured and learnable?
- where does it appear unstable, sparse, or tail-dominated?

### Key summary numbers

From [summary.json](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/summary.json):
- total samples: `7053`
- delta-bucket counts:
  - `flat`: `4403`
  - `mild_rise`: `987`
  - `mild_drop`: `789`
  - `sharp_rise`: `417`
  - `sharp_drop`: `387`
- test error by bucket:
  - `flat`: `0.274`
  - `mild_drop`: `0.479`
  - `mild_rise`: `0.518`
  - `sharp_drop`: `2.314`
  - `sharp_rise`: `2.982`

This is the most important learnable-space finding in the repo:
- the problem is much easier when EPS changes are small or moderate
- performance degrades sharply in the large-move buckets

### 1. `pca_separability.png` and `umap_separability.png`

Files:
- [pca_separability.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/pca_separability.png)
- [umap_separability.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/umap_separability.png)

What they show:
- a 2D embedding of the feature space
- color overlays for target buckets and test error

How to read them:
- points near one another are feature-similar under the chosen embedding
- PCA preserves large linear directions of variance
- UMAP preserves local neighborhood structure more than global linear axes

What to look for:
- whether `flat`, `mild_*`, and `sharp_*` events occupy distinct regions or heavily overlap
- whether high-error points cluster in specific parts of the manifold

Practical interpretation:
- if high-error sharp-move events cluster in sparse or isolated regions, the model is facing a true tail-distribution problem
- if they overlap heavily with ordinary events, then the features may simply be insufficient to disambiguate them

### 2. `pca_faceted_by_delta_bucket.png` and `umap_faceted_by_delta_bucket.png`

Files:
- [pca_faceted_by_delta_bucket.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/pca_faceted_by_delta_bucket.png)
- [umap_faceted_by_delta_bucket.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/umap_faceted_by_delta_bucket.png)

These are the clearest visual diagnostics.

What they show:
- one panel per `delta_bucket`
- the same embedding space in every panel
- the current bucket highlighted
- highlighted points colored by absolute test error
- panel titles include sample counts and proportions

How to read them:
- compare where each bucket lives in the embedding space
- compare how common each bucket is
- compare how much error each bucket carries

Main takeaway they support:
- `flat` and mild-move events are dense, common, and relatively low-error
- `sharp_rise` and `sharp_drop` are sparse and much higher-error

### 3. `*_company_specific_only.png`

Files:
- [pca_separability_company_specific_only.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/pca_separability_company_specific_only.png)
- [umap_separability_company_specific_only.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/umap_separability_company_specific_only.png)
- [pca_faceted_by_delta_bucket_company_specific_only.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/pca_faceted_by_delta_bucket_company_specific_only.png)
- [umap_faceted_by_delta_bucket_company_specific_only.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/umap_faceted_by_delta_bucket_company_specific_only.png)

These embeddings exclude the shared macro/context sequence features and keep only company-specific information.

How to interpret the difference versus the full plots:
- if structure becomes clearer, the shared macro features were obscuring company-level organization
- if structure disappears, the macro context was doing important work

In other words:
- `full` answers: what does the total predictive space look like?
- `company_specific_only` answers: what structure is intrinsic to the company/event itself?

### 4. `target_surfaces_actual_eps.png`

File:
- [target_surfaces_actual_eps.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/target_surfaces_actual_eps.png)

This plot family shows actual EPS against a handful of interpretable features.

Axes/features included:
- `last_prior_eps`
- `baseline_trailing_mean`
- `prev_eps_yoy_change`
- `revenue_growth_yoy`
- `operating_margin`
- `vix_close`

How to read it:
- look for smooth monotone structure versus diffuse clouds
- strong smooth structure suggests a feature is genuinely informative for EPS level
- weak or flat structure suggests limited direct explanatory power for EPS level

Main interpretation from the summary statistics:
- `baseline_trailing_mean` has the strongest direct relationship with actual EPS
  - correlation about `0.80`
- `last_prior_eps` is also strong
  - correlation about `0.74`
- the other variables are much weaker for raw EPS level

This is consistent with the experimental outcome that simple historical EPS baselines remain very strong.

### 5. `target_surfaces_delta_last.png`

File:
- [target_surfaces_delta_last.png](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/target_surfaces_delta_last.png)

This is often the more important surface plot, because the better target framing was the residual change from last EPS.

How to read it:
- you are no longer asking what explains absolute EPS level
- you are asking what explains deviations from the last known EPS

The strongest relationship in the summary is:
- `prev_eps_yoy_change` vs `delta_last`
  - correlation about `-0.55`

Interpretation:
- prior EPS change dynamics are much more informative for residual movement than macro or simple fundamentals alone
- this supports the choice of `delta_last` as the better target formulation

### 6. `analysis_frame.csv` and `embedding_coordinates.csv`

Files:
- [analysis_frame.csv](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/analysis_frame.csv)
- [embedding_coordinates.csv](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/embedding_coordinates.csv)

Use these if you want to reproduce or extend the visualizations.

They contain:
- event-level metadata
- bucket assignments
- selected interpretable features
- embedding coordinates for each sample

### 7. `sectors/`

Directory:
- [sectors/](/home/zach/wd7/projects/eps-estimation/artifacts/learnable_space_analysis/sectors)

Each sector subdirectory contains the same analysis suite for that sector alone.

How to use them:
- compare whether the same “easy near flat / hard at large moves” pattern holds within sectors
- identify sectors where the feature geometry is tighter or looser than the global view

This is useful because some sectors are structurally more stable than others.

## Practical Takeaway

The experiments support a fairly concrete interpretation:

- the project found real predictive structure
- the most reliable structure comes from prior EPS history
- large EPS jumps remain the hardest region of the problem
- trimming or excluding tail tickers helps a lot
- broader data and residual targets help
- the best learned model so far is the trimmed global LSTM point model
- the strongest simple benchmark remains the trailing-mean baseline
- uncertainty estimation remains under-calibrated and would benefit from calibration methods such as conformal calibration

If a reader should remember one sentence, it is this:
- the repo did not produce a universally superior EPS forecaster, but it did identify where the problem is learnable, what target framing works best, and which baselines are genuinely hard to beat.
