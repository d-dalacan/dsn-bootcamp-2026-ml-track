# DSN Bootcamp Qualification Hackathon 2026 — ML Track

Predicting product-store sales for DSN Mart, a retail chain across Nigeria, using product and store attributes. Built as a qualification submission for the DSN AI Bootcamp.

**Competition:** [DSN Bootcamp Qualification Hackathon 2026 — ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)
**Task:** Regression — predict `total_sales` for a given product at a given store
**Metric:** RMSE (lower is better)
**Best score:** 1076.43 RMSE (personal best on the public leaderboard)

## Problem

DSN Mart operates stores ranging from small corner shops to flagship hypermarkets across Nigeria. Given historical product-store sales records, the goal is to predict total sales for unseen product-store combinations — helping the business plan stock, pricing, and store investment.

## Approach

1. **Exploratory Data Analysis** — examined missing values, target distribution, and relationships between sales and store format, location tier, store size, and product price.
2. **Data Cleaning** — imputed missing `product_weight_kg` using each product's own median, and `store_size` using each store's own most common value, avoiding the ~28-46% data loss a naive drop would have caused.
3. **Feature Engineering** — ordinal encoding for naturally-ordered categories (store size, location tier), `price_per_kg`, and a leak-safe target-encoded average sales per store and per product category.
4. **Modeling** — progressed from a Linear Regression baseline, to Random Forest, to XGBoost, tuned via manual and grid search.
5. **Data quality fix** — discovered `product_category` had 48 raw values that were actually only 16 real categories, fragmented by inconsistent capitalization. Standardizing this and rebuilding with leak-safe, cross-validated target encoding gave the largest single improvement.
6. **Ensembling** — blended XGBoost and LightGBM predictions using 5-fold cross-validation.

## Results

| Round | Approach | Leaderboard RMSE |
|---|---|---|
| 1 | Random Forest baseline | 1107.37 |
| 2 | Tuned XGBoost + store-average feature | 1086.72 |
| 3 | Category bug fix + leak-safe encoding + 5-fold CV + XGBoost/LightGBM blend | **1076.43** |

**Key finding:** the single largest improvement came from a data-cleaning fix (the category capitalization bug), not from model choice or hyperparameter tuning — a reminder that data quality often matters more than model sophistication.

## What I'd try next

- Interaction features between store format and product category
- CatBoost, which handles categorical features natively
- A learned (stacked) blend instead of a flat average

## Tech stack

Python, pandas, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn

## Author

Biutrus Daniel Dalacan — 400-level Computer Science, Federal University of Technology, Minna

[Kaggle Profile]([https://www.kaggle.com/danieldalacan)
