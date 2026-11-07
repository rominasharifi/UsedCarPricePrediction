# Used Car Price Prediction using Machine Learning

A supervised regression project that predicts the price of a used car from its listing details, using Python, scikit-learn, and XGBoost. This was a team project (team of 5) for CSCI-6751 (Artificial Intelligence) at Fairleigh Dickinson University.

## Dataset

- Source: Kaggle Playground Series, Season 4 Episode 9, "Regression of Used Car Prices" (train set)
- 188,533 vehicle listings, 12 features after dropping the `id` column
- Target: `price`
- Features: `brand`, `model`, `model_year`, `milage`, `fuel_type`, `engine`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title`

The data is not included in this repository. Download `train.csv` from Kaggle and place it next to the notebooks.

## Workflow

1. **Data inspection:** loaded the data, checked types, and summarized the numeric columns.
2. **Missing values:** three categorical columns had missing values: `fuel_type` (5,083 rows, 2.7%), `accident` (2,452 rows, 1.3%), and `clean_title` (21,419 rows, 11.4%). Missing values were filled with an "Unknown" category.
3. **High-cardinality features:** `model` has 1,897 distinct values and `engine` has 1,117, so one-hot encoding was not practical for them. `brand` and `model` were combined into one feature, and target and frequency encoding were used for the high-cardinality columns.
4. **Outliers:** removed mileage outliers with the IQR method, applied separately for each model year (188,533 rows before, 183,867 after).
5. **Modeling:** trained Linear Regression and XGBoost regressors and compared them with 5-fold cross-validation, using MAE, RMSE, and R² as metrics.
6. **Hyperparameter tuning:** tuned XGBoost with `RandomizedSearchCV` (100 iterations, 3-fold cross-validation, MAE scoring), then refined the best parameters with `GridSearchCV`.

## Tools and Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- category_encoders, dython
- Google Colab

## My Contributions

- Data exploration and preprocessing (missing values, high-cardinality analysis, outlier removal), together with a teammate
- Presented the dataset overview, exploratory analysis, and preprocessing sections in the final class presentation

## Files

- `ai_project.ipynb`: data loading, exploratory analysis, and preprocessing
- Modeling notebook: Linear Regression, XGBoost, and hyperparameter tuning

## Future Work

- Add more informative features, such as service history and region
- Compare additional models, such as LightGBM
- Evaluate on a separate hold-out set after tuning
