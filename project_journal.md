# Project Journal — ANL488 FYP
## Factory Price Prediction for SNRE

This is a running record of my thought process throughout the project — decisions made, dead ends hit, and things I changed my mind about. Written for myself, not for an audience.

---

## Understanding the Problem

My project sponsor is SNRE, a company in Singapore's industrial real estate market. The ask was simple on the surface: build a model to predict factory unit prices ($ psf).

But I had to ask — why? Two reasons emerged:
1. Manual valuations are slow and labour-intensive for agents
2. Buyers have no independent benchmark to know if they're paying a fair price

That second one stuck with me. The model isn't just an operational tool — it's a transparency tool. That framing influenced every decision I made about which features to include and how to present results.

The target variable is unit price per square foot ($ psf), not total transaction price. This is intentional — it normalises for factory size so properties of different sizes are comparable.

---

## EDA — What the Data Actually Told Me

I went into EDA expecting macro features (GDP, CPI, interest rates) to be important signals. I was wrong, but I didn't know that yet.

**What surprised me most:**

`Remaining_Lease_Years` had a correlation of r=+0.672 with unit price — by far the strongest predictor. This makes intuitive sense in retrospect: factory buyers are essentially buying time on the land. A property with 6 years of lease left is fundamentally different from one with 55 years, even if everything else is identical.

`Area (sqft)` had a negative correlation (r=-0.377). This is a bulk discount effect — buyers of large units pay less per sqft because the absolute deal size is already large. I needed to log-transform this later to linearise the relationship.

Central Region commanded a 70% price premium over West Region. Kallang specifically traded at $1,050 psf median — roughly double the dataset median of $435 psf. Location matters enormously in Singapore industrial property.

The macro features were suspicious from the start. With only 3 years of data (2023–2025), there are just 12 unique quarterly values for GDP, CPI, unemployment etc. Every transaction in the same quarter shares identical macro values. That's not really signal — it's more like a time label. I flagged this early but decided to retain them and test empirically later.

**The target variable:**

Skewness of +0.82 with a long right tail. This matters because linear regression assumes normally distributed residuals — a skewed target breaks that assumption. I knew from the start I'd need to log-transform the target for linear models.

---

## Feature Engineering — Decisions and Why

### Log-transforming the target

This is about the model's errors, not the values themselves. When the target is skewed, linear regression overfits the dense $200–$600 psf range and systematically underpredicts high-value properties. Log-transform fixes the residual distribution so the model is equally accountable across the full price range.

One thing I had to get clear in my head: log-transformed values are not interpreted directly. A predicted value of 6.1 means `exp(6.1) ≈ $445 psf`. The log scale is internal to the modelling process — stakeholders always see dollar values after back-transformation.

### Log-transforming Area (sqft)

Different reason from the target transform. Area had a curved relationship with price in the scatter plot — a steeply declining bulk discount that flattens for large units. Linear regression needs straight-line relationships. Log-transforming area linearises that curve.

Key distinction I kept coming back to:
- Log target → fixes residual distribution
- Log feature → fixes linearity assumption

### Lease Remaining Ratio

`Remaining_Lease_Years / Lease_Duration`

I added this because a property with 30 years remaining on a 60-year lease (50% consumed) is fundamentally different from one with 30 years on a 99-year lease (70% remaining). Neither raw feature captures this proportional picture alone.

My worry was collinearity — the ratio is derived from two existing features. I ran VIF before and after:
- `Remaining_Lease_Years` VIF = 1.45
- `Lease_Remaining_Ratio` VIF = 1.41

Both well below 2.0. The ratio adds genuinely independent information because `Lease_Duration` (which gets dropped after the ratio is computed) carries the context that the ratio needs.

### MRT Distance — a late addition

Partway through the project, a request came in to add distance to the nearest MRT station as a feature. This required:
1. Geocoding 148 unique factory buildings via OneMap API
2. Loading 193 MRT/LRT station coordinates
3. Computing Haversine distance (great-circle distance) from each factory to its nearest station

The result: r = -0.316 with unit price. Factories closer to MRT command higher prices — consistent with how Singapore real estate behaves generally.

After model training, `dist_to_mrt_m` turned out to be the 4th most important feature (8.1% importance), behind remaining lease (29.6%), planning area encoding (24.0%), and log area (12.1%). Adding it was worth it.

### What I decided NOT to do

**PCA on macro features**: I initially listed PCA as a feature engineering step. Threw it out. VIF had already resolved the multicollinearity problem — applying PCA on top would reduce interpretability for zero additional benefit. Over-engineering.

**Time-based train/test split**: I considered this since the data spans 2023–2025. But this is a cross-sectional regression, not a time-series forecast. I'm not trying to predict future prices — I'm predicting the price of a property given its characteristics. A random 80/20 split is appropriate. The macro features carry whatever time signal exists.

---

## Multicollinearity — More Serious Than I Expected

I ran VIF on all features and found two severe cases:
- `Price_Index`: VIF = 88.14 (98.9% of its variance explained by other features)
- `Steel_Rebar_Per_Tonne`: VIF = 38.14 after removing Price_Index

These had to go. A VIF of 88 means the coefficient is fitted almost entirely on noise — it becomes unstable and uninterpretable.

The analogy that made this click for me: imagine trying to figure out which of two twins ate the cake when they were always in the same room together. You can measure their combined effect, but you can never isolate individual contributions. That's exactly the multicollinearity problem.

Tree models are immune to this — they split on one feature at a time and don't need to isolate individual effects. But for Ridge regression, multicollinearity causes coefficient explosion and makes the model unpredictable.

---

## Model Selection — Why These Three

I chose Ridge, Random Forest, and XGBoost. Each served a specific purpose:

**Ridge**: interpretable baseline. Not because I expected it to win, but because I needed a model where I could look at the coefficients and say "each additional lease year increases price by X%." SNRE stakeholders think in terms of factors, not black-box predictions.

**Random Forest**: non-linear benchmark. Lease and location effects are non-linear in ways Ridge can't fully capture even with log-transforms. Random Forest handles this naturally.

**XGBoost**: performance ceiling. Gradient boosting typically outperforms bagging on tabular data. If Random Forest was already good, XGBoost would be better — and it was.

I didn't need more models. The question was answered by three.

### Why I retained macro features longer than maybe I should have

The honest reason: I wasn't sure. With only 12 quarterly data points, the signal-to-noise ratio was low — but low signal isn't the same as no signal. Economic cycles affect property prices. I decided to let the model tell me, rather than dropping them based on intuition.

---

## Results — What Surprised Me

| Model | RMSE | R² |
|-------|------|-----|
| Ridge | $82.60 $/psf | 0.7845 |
| Random Forest | $48.41 $/psf | 0.9260 |
| XGBoost | $46.82 $/psf | 0.9307 |

The gap between Ridge and the tree models was larger than I expected — R² of 0.78 vs 0.93. This tells me the lease and location relationships are more non-linear than log-transforming could fix. Ridge was useful as an explanation tool, but not as a prediction tool.

The RMSE vs MAE gap was also informative. Ridge had a gap of $21.56; XGBoost had $14.05. A smaller gap means fewer catastrophic outlier predictions. XGBoost handled edge cases better.

### The macro ablation test

After notebook 09 showed macro features collectively accounting for only 4.9–8.3% of importance, I ran a proper ablation test — retrained XGBoost without all 6 macro features and compared RMSE.

Result: removing macro features improved RMSE by $2.12 (from $46.82 to $44.70).

This was the decision-making moment I'm most satisfied with. I didn't drop them because the importance was low. I tested empirically, got clear evidence, and then dropped them. The model was actually learning spurious quarterly patterns from the macro features — patterns that didn't generalise.

The lesson: low feature importance is a hypothesis, not a conclusion. Test it.

---

## Preprocessing Pipeline Design

One decision I thought carefully about: where to put the preprocessing logic.

The preprocessing — log-transforming area, computing the lease ratio, target-encoding planning area, one-hot encoding categoricals — needed to run consistently during both training (notebook 05) and inference (serving predictions to SNRE). If the logic lives in two places, it will eventually diverge and produce wrong predictions.

Solution: notebook 05 defines the logic and saves the fitted encoders (`target_encoder.pkl`, `onehot_encoder.pkl`). `src/inference.py` loads those same encoders and applies the same transforms at inference time. One source of truth.

`Lease_Duration` is an interesting case. It's an input to the inference pipeline (needed to compute the ratio) but not a model feature. The user provides it, we compute the ratio, then discard it before passing to the model.

---

## Deployment — Building Something SNRE Can Actually Use

A model in a `.pkl` file is not a product. I built a working demo with:
- **FastAPI backend** (`backend/api.py`) — serves predictions via HTTP POST
- **Streamlit frontend** (`frontend/app.py`) — form-based UI for SNRE agents
- **Shared inference logic** (`src/inference.py`) — preprocessing + prediction, independent of how it's served

The separation matters: if SNRE later wants a proper web app instead of Streamlit, only the frontend changes. The backend and inference logic stay untouched.

**One bug I hit**: the first version loaded model artifacts from disk on every request. The first prediction call timed out because loading XGBoost takes several seconds. Fixed by caching artifacts in memory on first load — every subsequent request hits the cache and responds in milliseconds. Simple fix, but it mattered.

**Input validation via Pydantic**: I separated validation into `backend/models.py`. The API rejects invalid inputs before they reach the model — unknown planning areas, negative floor areas, remaining lease exceeding total duration. Garbage in, garbage out is a real problem in production.

---

## What I'd Do Differently

**Longer macro history**: with 10 years of data instead of 3, the macro features might have shown genuine signal. With only 12 quarterly values, there was never enough variation to learn real economic relationships.

**Walking-time distance to MRT**: I used straight-line Haversine distance. This ignores roads, expressways, and barriers. A factory 300m from an MRT station but separated by an expressway is very different from one 300m away with a direct footpath. Walking-time isochrones (from OneMap or Google Maps API) would be more accurate.

**Prediction intervals**: the model returns `predicted ± RMSE` as a rough range. This is not a true confidence interval — RMSE is an average, not a per-prediction measure. Quantile regression or bootstrapped prediction intervals would give SNRE a more accurate sense of uncertainty per property.

**Separate models per lease segment**: 30-year, 60-year, and 99-year leases may have different price dynamics. A single model trained across all three may not capture segment-specific patterns well.

---

## Key Numbers I Need to Remember

| Metric | Value |
|--------|-------|
| Dataset size | 3,782 transactions |
| Date range | Jan 2023 – Dec 2025 |
| Target median | $435 psf |
| Strongest predictor | Remaining_Lease_Years (r=+0.672) |
| MRT distance importance | 4th (8.1% avg importance) |
| Macro features combined importance | 4.9–8.3% |
| Macro ablation RMSE improvement | +$2.12 $/psf |
| Champion model | XGBoost (reduced, no macro) |
| Champion RMSE | $44.70 $/psf |
| Champion R² | 0.9307 |
