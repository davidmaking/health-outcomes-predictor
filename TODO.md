# Project Checklist

## Data Merging
- [x] Load and clean EPA Smart Location Database
- [x] Load and clean CDC PLACES (obesity, diabetes)
- [x] Pull Census ACS demographics (MHI, race, no-vehicle ratio)
- [x] Merge EPA + PLACES + Census into `df`
- [x] Load USDA Food Access Research Atlas and select key variables (`food_slim`)
- [x] Merge USDA Food Access Research Atlas into `df`

## Data Cleaning
- [x] Fix GEOID precision loss (scientific notation in EPA CSV)
- [x] Aggregate EPA block groups → census tracts
- [x] Pivot PLACES from long to wide format
- [x] Drop rows missing key outcome/control variables
- [x] Remove Census MHI sentinel values (-666666666)
- [x] Deduplicate on GEOID_tract

## EDA
- [x] Summary statistics on key variables
- [x] Distribution histograms
- [x] Correlation heatmap
- [x] Bivariate scatter plots (walkability, income, no-vehicle ratio vs. outcomes)
- [x] State-level variation charts
- [x] EDA on food desert variables after merge (describe + correlation)
- [x] Check for regional patterns / outlier tracts

## Modeling — Week 3
- [x] OLS regression: obesity ~ walkability + controls
- [x] OLS regression: diabetes ~ walkability + controls
- [x] Add food desert indicator and compare model fit (R², coefficients)
- [x] Check statistical significance and confidence intervals
- [x] Test interaction: food desert × income, food desert × no-vehicle ratio

## Modeling — Week 4
- [x] Random Forest: predict obesity/diabetes from all features
- [x] Feature importance plot
- [x] Identify underperforming tracts (actual vs. predicted residuals)
- [x] Map or cluster underperforming tracts — geographic confounding?
- [x] Subgroup analysis: low-income tracts, low car-ownership tracts

## Modeling - Week 5
- [ ]
- [ ]
- [ ]
- [ ]

## Deliverables — Week 5
- [ ] Final slide deck
- [ ] Clean up and comment notebook
- [ ] Push to GitHub with README
