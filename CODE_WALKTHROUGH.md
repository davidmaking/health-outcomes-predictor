# Notebook Code Walkthrough

A plain-language explanation of every code block in `analysis.ipynb`, in order.

---

## Section 0 — Imports & Config (Cell 2)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
...
DATA_DIR = '../data/'
```

Loads the tools the notebook depends on:
- **pandas** — for loading and manipulating tabular data (like Excel, but in Python)
- **numpy** — for math operations on arrays of numbers
- **matplotlib / seaborn** — for making charts and plots

Also sets two display preferences (show up to 50 columns, format decimals to 4 places) and stores the path to the data folder as a variable so it doesn't need to be retyped everywhere.

---

## Section 1 — Load Data (Cells 4–5)

**Cell 4 — Load EPA data**
```python
epa = pd.read_csv(EPA_FILE, dtype={'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': str})
```
Reads the EPA walkability CSV into a table called `epa`. The `dtype` argument forces the three geographic ID columns (state, county, tract codes) to be read as text rather than numbers. This matters because leading zeros would be dropped if they were read as integers — e.g., state code `"01"` (Alabama) would silently become `1`.

**Cell 5 — Load CDC PLACES data**
```python
places = pd.read_csv(PLACES_FILE, dtype={'LocationID': str})
```
Reads the CDC PLACES health outcomes CSV. Same idea — `LocationID` is a census tract identifier that must stay as text.

---

## Section 2 — Explore Schemas (Cells 7–9)

**Cell 7** prints every column name and its data type in the EPA table.

**Cell 8** does the same for PLACES.

**Cell 9** prints the list of unique health measures in PLACES (e.g., `OBESITY`, `DIABETES`, `BPHIGH`, etc.) so we can see what's available and confirm the columns we need exist.

---

## Section 3 — Clean EPA Data (Cell 11–12)

**Cell 11 — Build tract GEOID and aggregate**
```python
epa['GEOID_tract'] = epa['STATEFP'].str.zfill(2) + epa['COUNTYFP'].str.zfill(3) + epa['TRACTCE'].str.zfill(6)

epa_tract = epa.groupby('GEOID_tract').agg(
    walkability_mean=('NatWalkInd', 'mean'),
    ...
)
```
The EPA data is at the **block group** level — smaller than census tracts. We need tract-level data to match the health outcomes. This cell:
1. Builds an 11-digit tract ID by concatenating the state (2 digits), county (3 digits), and tract (6 digits) codes. `zfill` pads with leading zeros to ensure the correct length.
2. Groups all block groups that share the same tract ID and computes the mean, min, and max walkability score for that tract.

**Cell 12** checks whether any tracts have missing walkability values after the aggregation.

---

## Section 4 — Clean PLACES Data (Cells 14–16)

**Cell 14 — Filter to obesity and diabetes**
```python
target_measures = ['OBESITY', 'DIABETES']
places_filtered = places[places['MeasureId'].isin(target_measures)].copy()
```
PLACES contains dozens of health metrics. This keeps only the two we care about.

**Cell 15 — Pivot from long to wide format**
```python
places_wide = places_filtered.pivot_table(
    index='LocationID', columns='MeasureId', values='Data_Value', aggfunc='first'
)
```
The original table has one row per measure per tract — so each tract appears twice (once for obesity, once for diabetes). This pivot reshapes it so each tract has a single row with separate `obesity_pct` and `diabetes_pct` columns.

**Cell 16** checks for missing values in the reshaped table.

---

## Section 5 — Merge EPA + PLACES (Cells 18–19)

**Cell 18**
```python
df = epa_tract.merge(places_wide, on='GEOID_tract', how='inner')
```
Joins the EPA walkability table and the PLACES health outcomes table on the shared census tract ID. `how='inner'` means only tracts that appear in **both** datasets are kept.

**Cell 19** drops any rows where obesity or diabetes values are still missing after the merge, since those tracts can't be used in the analysis.

---

## Section 6 — Early EDA (Cells 21–23)

**Cell 21** prints basic summary statistics (min, max, mean, percentiles) for walkability, obesity, and diabetes.

**Cell 22** computes the correlation between those three variables — a number between -1 and 1 showing how strongly they move together.

**Cell 23** draws two scatter plots (walkability vs obesity, walkability vs diabetes) to visually inspect the relationship before any modeling.

---

## Section — Load Food Access Data (Cells 24–28)

**Cell 24–25 — Load USDA food access file**
```python
FOOD_FILE = pd.read_csv(DATA_DIR + 'food_access.csv')
food = FOOD_FILE
```
Reads the USDA Food Access Research Atlas CSV, which classifies census tracts as food deserts based on distance to grocery stores and income levels.

**Cell 26** identifies columns with many missing values (more than 1,000 nulls) so we know what to avoid.

**Cell 28 — Select relevant columns**
```python
food_vars = ['CensusTract', 'LILATracts_1And10', 'LILATracts_halfAnd10', ...]
food_slim = food[food_vars].copy()
food_slim['CensusTract'] = food_slim['CensusTract'].astype(str).str.zfill(11)
```
Keeps only the ~12 columns relevant to the research questions. The USDA dataset has 140+ columns. The `zfill(11)` again ensures the tract ID has the right number of digits for joining. Key columns kept:
- `LILATracts_1And10` — primary food desert flag (low income + low access)
- `lapop1share` — share of tract population with low food access within 1 mile
- `lahunv1share` — share of households with no car that also have low food access
- `PovertyRate`, `MedianFamilyIncome` — cross-checks against Census data

---

## Section — Pull Census Demographics (Cells 29–33)

**Cell 29** connects to the US Census Bureau API using a pre-set API key.

**Cell 30 — Download ACS demographic data**
```python
for state_fips in c.acs5.get('NAME', {'for': 'state:*'}):
    data = c.acs5.state_county_tract(variables, fips, '*', '*')
```
Loops through every state and downloads American Community Survey (ACS) data for every census tract. The variables pulled are: median household income, number of zero-car households, racial population counts, and total population/households.

**Cell 31** renames the raw Census variable codes (like `B19013_001E`) to readable names like `MHI` (median household income).

**Cell 32** converts raw population counts into percentages by dividing each group's count by the total population.

**Cell 33** renames the percentage columns to cleaner names and builds the final demographics table with race percentages, no-vehicle ratio, and the tract ID.

---

## Section — Merge Demographics (Cell 35)

```python
df = df.merge(DEMOGRAPHICS, left_on='GEOID_tract', right_on='TRACT', how='left')
```
Adds the Census demographic columns (income, race, car ownership) to the main `df` table. `how='left'` keeps all tracts already in `df` even if a demographic match isn't found.

---

## Section 7 — Final Data Cleaning (Cells 37–41)

**Cell 37** prints how many missing values exist in each key column after all merges, so we know the extent of data loss.

**Cell 38 — Drop rows with missing controls**
```python
df = df.dropna(subset=key_cols)
```
Removes any tract that is missing income, race, or car-ownership data. These control variables are required for all regressions, so rows without them can't be used.

**Cell 39 — Rename columns**
```python
df = df.rename(columns={'NHW PERCENT': 'nhw_pct', ...})
```
Shortens column names so they're easier to type in formulas and plot labels.

**Cell 40 — Remove Census sentinel values**
```python
df = df[df['MHI'] >= 0]
```
The Census API uses `-666666666` as a placeholder for suppressed or unavailable income data. This removes those rows since a negative income is impossible and would corrupt any analysis.

**Cell 41 — Remove duplicate tracts**
```python
df = df.drop_duplicates(subset='GEOID_tract')
```
Ensures each census tract appears only once in the final dataset.

---

## Section 8a — Summary Statistics (Cell 43)

```python
df[key_cols_new].describe().round(2)
```
Prints a table showing count, mean, standard deviation, min, 25th/50th/75th percentiles, and max for all key variables. A quick sanity check that values are in a plausible range.

---

## Section 8b — Distribution Histograms (Cell 45)

```python
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, (col, label) in zip(axes.flat, plot_vars):
    ax.hist(df[col].dropna(), bins=60, ...)
```
Draws 8 histograms in a 2×4 grid — one for each key variable. Each histogram shows how values are spread across all census tracts, making it easy to spot skewed distributions or unusual values.

---

## Section 8c — Correlation Heatmap (Cell 47)

```python
corr = df[corr_cols].corr().round(2)
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ...)
```
Computes pairwise correlations between all key variables and displays them as a color-coded grid. Blue = positive correlation, red = negative. Numbers in each cell are the correlation coefficient. This lets you quickly see which predictors are most related to health outcomes and which predictors are correlated with each other (which can cause problems in regression).

---

## Section 8d — Bivariate Scatter Plots (Cell 49)

```python
pairs = [('walkability_mean', 'obesity_pct', ...), ...]
for ax, (x, y, title) in zip(axes.flat, pairs):
    ax.scatter(df[x], df[y], alpha=0.05, s=4)
    m, b = np.polyfit(...)
    ax.plot(xs, m * xs + b, color='red', ...)
```
Draws 6 scatter plots comparing each predictor (walkability, income, no-vehicle ratio) against each outcome (obesity, diabetes). The red line is a simple linear trend line fitted to the data.

---

## Section 8e — State-Level Variation (Cells 51–52)

**Cell 51**
```python
df['state_fips'] = df['GEOID_tract'].str[:2]
state_obesity = df.groupby('state_fips')['obesity_pct'].agg(['mean', 'std', 'count'])
```
Extracts the 2-digit state code from each tract's GEOID, then computes mean obesity by state. Plots a horizontal bar chart highlighting the 10 most obese and 10 least obese states.

**Cell 52** does the same for mean walkability score by state.

---

## Section — Merge Food Access Data (Cell 53)

```python
df = df.merge(food_slim, left_on='GEOID_tract', right_on='CensusTract', how='left')
```
Adds the USDA food access columns to the main `df`. Happens after the state-level EDA because the merge adds columns that need to be checked separately.

**Cell 54** runs `df.describe()` as a final sanity check on all columns including the new food access ones.

---

## Section — Full Correlation Heatmap with Food Access (Cell 55)

Same as the earlier heatmap but now includes the food access variables (`LILATracts_1And10`, `lapop1share`, `lalowi1share`, `lahunv1share`). Overwrites the previous heatmap PNG with the updated version.

---

## Section 8f — Regional Patterns & Outlier Tracts (Cells 57–67)

**Cell 57 — Assign Census regions**
```python
region_map = {'09': 'Northeast', '17': 'Midwest', ...}
df['region'] = df['state_fips'].map(region_map)
```
Uses a lookup dictionary to assign each tract to one of four Census regions (Northeast, Midwest, South, West) based on its state FIPS code.

**Cell 58** draws box plots comparing obesity and diabetes distributions across the four regions.

**Cell 59** draws a similar box plot for walkability by region.

**Cell 60** computes a summary table with mean and median of walkability, health outcomes, income, and food desert prevalence for each region.

**Cell 62 — Z-score outlier detection**
```python
z = df[z_cols].apply(stats.zscore, nan_policy='omit')
outlier_mask = (z.abs() > 3).any(axis=1)
```
Converts each variable to a z-score (how many standard deviations a value is from the mean). Flags any tract that is more than 3 standard deviations from the mean on any variable. These are statistical outliers that may represent data errors or genuinely unusual neighborhoods.

**Cell 63** lists the 10 tracts with the lowest and highest walkability z-scores.

**Cell 64** lists the 10 tracts with the highest obesity and diabetes rates.

**Cell 65 — Surprising tracts**
```python
surprising = df_z[(df_z['walkability_mean_z'] > 1) & (df_z['obesity_pct_z'] > 1.5)]
```
Finds tracts that are above average on walkability but also well above average on obesity. These are counterintuitive — walking-friendly but still unhealthy — and worth investigating for confounders.

**Cell 66** redraws the walkability vs health scatter plots with outlier tracts highlighted in red.

**Cell 67 — Food desert prevalence by region**
Computes what percentage of tracts in each Census region are classified as food deserts, then plots it as a bar chart.

---

## Section 9 — Regression Modeling — Week 3

### 9a. Baseline OLS Models (Cells 69–70)

```python
controls = 'MHI + nhw_pct + black_pct + hispanic_pct + no_vehicle_ratio'
ols_ob_base = smf.ols(f'obesity_pct ~ walkability_mean + {controls}', data=df).fit(cov_type='HC3')
```
Runs two Ordinary Least Squares (OLS) regressions — one for obesity, one for diabetes. The formula says: "predict the outcome using walkability plus income, race percentages, and car ownership as controls." `cov_type='HC3'` uses robust standard errors, which are more reliable when the variance of errors isn't constant across tracts (which is common in geographic data).

Reading the output: the coefficient on `walkability_mean` tells you how much the outcome changes on average for each 1-point increase in walkability, holding everything else constant.

### 9b. Add Food Desert Indicator (Cells 72–75)

**Cells 72–73** re-run the same regressions but add `LILATracts_1And10` (the food desert flag) as an additional predictor.

**Cell 74 — Model comparison table**
```python
comparison = pd.DataFrame({'R²': [...], 'Adj R²': [...], 'AIC': [...]})
```
Puts all four models side by side with their R² (how much variance the model explains), Adjusted R² (penalizes for adding more predictors), and AIC (lower = better fit). This directly answers research question 3: does adding food desert status improve the model?

**Cell 75 — Coefficient plot**
Draws a horizontal bar chart for each model where each bar is a predictor's coefficient with whiskers showing the 95% confidence interval. Bars that don't cross zero are statistically significant. Blue = positive effect on the outcome, red = negative.

### 9c. Interaction Terms (Cells 77–78)

**Cell 77 — Food desert × income interaction**
```python
smf.ols('obesity_pct ~ walkability_mean + LILATracts_1And10 * MHI + ...', data=df)
```
The `*` creates an interaction term between food desert status and income. The coefficient on `LILATracts_1And10:MHI` answers: "does being in a food desert hurt more (or less) when income is lower?" A negative interaction coefficient would mean the food desert penalty is smaller at higher incomes.

**Cell 78 — Food desert × no-vehicle ratio interaction**
Same idea, but tests whether the food desert effect is amplified when residents don't have cars (which would make it physically harder to travel to distant grocery stores).

---

## Section 10 — Random Forest — Week 4

### 10a. Train Random Forest Models (Cell 80)

```python
rf_ob = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=30, ...)
rf_ob.fit(X_scaled, y_ob)
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
```
Random Forest builds hundreds of decision trees on random subsets of the data and averages their predictions. It's more powerful than linear regression because it can capture non-linear relationships and automatic interactions between predictors.

Key settings:
- `n_estimators=300` — build 300 trees and average them
- `max_depth=10` — each tree can split at most 10 levels deep (prevents overfitting)
- `min_samples_leaf=30` — each leaf node must contain at least 30 tracts (prevents overfitting on tiny groups)

`StandardScaler` normalizes each feature to have mean 0 and standard deviation 1, which helps the algorithm treat all predictors on equal footing.

`cross_val_score` with `cv=5` splits the data into 5 folds, trains on 4, tests on 1, and repeats — giving a more honest R² estimate than training and testing on the same data.

### 10b. Feature Importance Plot (Cell 82)

```python
imp = pd.Series(model.feature_importances_, index=features).sort_values()
imp.plot(kind='barh', ...)
```
Each feature gets an importance score showing how much it contributed to reducing prediction error across all 300 trees. Higher = more useful. This directly answers which predictor matters most: walkability, food desert status, income, or race.

### 10c. Residual Analysis (Cells 84–85)

**Cell 84 — Actual vs predicted plot**
```python
df_res['ob_resid'] = df_res['obesity_pct'] - df_res['ob_pred']
```
Computes the residual (actual minus predicted) for each tract. Scatter plots the predicted vs actual values — points on the diagonal red line are perfect predictions; points far above it are tracts the model severely underestimated.

**Cell 85 — Underperforming tracts**
```python
underperform_ob = df_res.nlargest(20, 'ob_resid')
```
Lists the 20 tracts where the model underestimated obesity the most (actual is much higher than predicted). Then checks which states those tracts cluster in — geographic clustering would suggest there's a regional factor the model is missing.

### 10d. Subgroup Analysis (Cells 87–88)

**Cell 87 — Random Forest by subgroup**
```python
subgroups = {
    'Low-income tracts (MHI < 25th pct)': df_model[df_model['MHI'] <= mhi_low_thresh],
    ...
}
for label, subset in subgroups.items():
    scores = cross_val_score(rf_sub, X_sub, y_sub, cv=5, ...)
```
Splits the data into four subgroups (bottom-quartile income tracts, all others, high no-car tracts, all others) and trains a separate Random Forest on each. Compares R² across groups to see whether the model predicts better or worse in economically disadvantaged or car-dependent areas.

**Cell 88 — OLS walkability coefficient by subgroup**
```python
m = smf.ols(f'{outcome} ~ walkability_mean + {controls}', data=subset).fit(cov_type='HC3')
subgroup_ols.append({'Walk β': m.params['walkability_mean'], 'p-value': ...})
```
For each subgroup, runs a standard OLS regression and extracts just the walkability coefficient (`β`). The resulting table shows whether walkability has a stronger or weaker association with health in low-income vs higher-income tracts — directly addressing the research question about whether income moderates the walkability effect.
