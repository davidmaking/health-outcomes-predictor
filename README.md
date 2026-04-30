# Walkability, Food Access & Public Health Outcomes

Does living in a walkable neighborhood lead to better health? Does living in a food desert make it worse? This project analyzes those questions at the US census tract level, using regression and machine learning to isolate the effects of walkability and food access on obesity and diabetes rates after controlling for income and race.

## Data

Four public datasets joined on census tract GEOID:

| Dataset | Description |
|---|---|
| [EPA Smart Location Database](https://www.epa.gov/smartgrowth/smart-location-mapping) | Walkability scores at the block group level |
| [CDC PLACES 2025](https://www.cdc.gov/places) | Tract-level obesity and diabetes prevalence estimates |
| [USDA Food Access Research Atlas](https://www.ers.usda.gov/data-products/food-access-research-atlas/) | Food desert classifications and low-access population shares |
| [Census ACS 5-year](https://www.census.gov/programs-surveys/acs) | Median household income, race/ethnicity, zero-car household ratio |

## Methods

- **Cleaning:** aggregated EPA block groups to tract level, pivoted CDC PLACES from long to wide, removed Census sentinel values
- **EDA:** distribution plots, correlation heatmaps, regional comparisons, outlier detection
- **Regression:** OLS, Ridge, LASSO, and ElasticNet with holdout (80/20) cross-validation; interaction terms for food desert × income and food desert × car ownership
- **Machine learning:** Random Forest with feature importance, residual analysis, and subgroup analysis by income and car ownership

## Web App

`app.py` is an interactive Streamlit app with two tabs:

- **Neighborhood Lookup** — enter any US address to pull up that census tract's walkability score, obesity rate, diabetes rate, and median income, each shown as a national percentile
- **What-If Explorer** — adjust walkability, food access share, and no-vehicle ratio (with income and race as controls) to see a live Gradient Boosting prediction of the diabetes rate

To run: `streamlit run app.py`

## Stack

Python — pandas, scikit-learn, statsmodels, matplotlib, seaborn, streamlit
