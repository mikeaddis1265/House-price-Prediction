# California Housing Price Prediction

A machine learning project to predict median house values in Californian districts based on various features such as location, housing characteristics, and demographic information.

## Project Overview

This project uses the California Housing dataset to build a regression model that predicts median house values. The model is trained on data from the 1990 California census and considers features like average income, population, location coordinates, and ocean proximity.

## Dataset

The dataset contains 20,640 entries with the following features:

- longitude: Longitude coordinate of the district
- latitude: Latitude coordinate of the district
- housing_median_age: Median age of houses in the district
- total_rooms: Total number of rooms in the district
- total_bedrooms: Total number of bedrooms in the district
- population: Total population in the district
- households: Total number of households in the district
- median_income: Median income of households in the district
- ocean_proximity: Proximity to ocean (categorical: <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN)
- median_house_value: Target variable - median house value in the district

## Installation

Install required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn scipy joblib
```

For API deployment (optional):
```bash
pip install fastapi uvicorn nest-asyncio pyngrok
```

## Project Structure

The project follows a standard machine learning workflow:

1. Data Loading and Exploration
2. Data Visualization
3. Train-Test Split with Stratified Sampling
4. Data Preprocessing and Feature Engineering
5. Model Training and Evaluation
6. Hyperparameter Tuning
7. Model Deployment

## Key Features

### Data Preprocessing

- Missing value imputation using median strategy
- Feature scaling with StandardScaler and MinMaxScaler
- One-hot encoding for categorical variables (ocean_proximity)
- Custom feature engineering:
  - rooms_per_house ratio
  - bedrooms_ratio
  - people_per_house ratio

### Feature Engineering

- Log transformation for skewed features
- Cluster-based similarity features using K-Means
- RBF kernel similarity to important locations
- Geographic clustering for location-based features

### Models Evaluated

1. Linear Regression
   - RMSE: 68,972.89

2. Decision Tree Regressor
   - Training RMSE: 0.0 (overfitting)
   - Cross-validation RMSE: 66,573.73 (+/- 1,103.40)

3. Random Forest Regressor (Best Model)
   - Cross-validation RMSE: 47,038.09 (+/- 1,021.49)
   - Final Test RMSE: 41,445.53
   - 95% Confidence Interval: [39,520.96, 43,701.77]

### Hyperparameter Tuning

Two approaches used:

1. Grid Search CV
   - Best parameters: n_clusters=15, max_features=6
   - Best score: -43,589.52

2. Randomized Search CV
   - Best parameters: n_clusters=45, max_features varies
   - Best score: -42,213.60

## Feature Importance

Top 5 most important features:

1. median_income (log transformed): 18.6%
2. ocean_proximity_INLAND: 7.3%
3. bedrooms_ratio: 6.6%
4. rooms_per_house_ratio: 5.4%
5. people_per_house_ratio: 4.6%

## Usage

### Training the Model

Run the Jupyter notebook to train the model:

```python
# Load and preprocess data
housing = load_housing_data()

# Train model
final_model = rnd_search.best_estimator_

# Save model
import joblib
joblib.dump(final_model, "my_california_housing_model.pkl")
```

### Making Predictions

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("my_california_housing_model.pkl")

# Prepare new data
new_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [880.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
})

# Predict
predictions = model.predict(new_data)
print(f"Predicted house value: ${predictions[0]:,.2f}")
```

## Model Pipeline

The final model uses a comprehensive preprocessing pipeline:

1. Numerical features:
   - Median imputation
   - Log transformation for selected features
   - Standard scaling

2. Categorical features:
   - Most frequent imputation
   - One-hot encoding

3. Custom transformations:
   - Ratio features
   - Cluster similarity features
   - Geographic features

4. Random Forest Regressor with optimized hyperparameters

## Results

- Final RMSE on test set: 41,445.53
- Model explains approximately 86% of variance in house prices
- Most influential factor: median income (log scale)
- Geographic features (cluster similarity) contribute significantly
- Ocean proximity has moderate impact, with INLAND showing highest importance

## API Deployment (Optional)

The project includes FastAPI implementation for model serving:

```python
# Run the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

API endpoint:
- POST /predict - Submit housing data and receive price prediction

Example request:
```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 20,
  "total_rooms": 1000,
  "total_bedrooms": 200,
  "population": 800,
  "households": 300,
  "median_income": 8.3,
  "ocean_proximity": "NEAR BAY"
}
```

## Data Source

Dataset source: https://github.com/ageron/data/raw/main/housing.tgz

Originally from: 1990 California Census

## Technologies Used

- Python 3.x
- pandas - Data manipulation
- NumPy - Numerical computing
- scikit-learn - Machine learning
- matplotlib - Data visualization
- scipy - Statistical functions
- joblib - Model serialization
- FastAPI - API framework (optional)

## Model Performance Notes

- The model shows good generalization with consistent performance across cross-validation folds
- Bootstrap confidence interval provides statistical confidence in test set performance
- Feature engineering significantly improved model performance over baseline
- Random Forest performed better than simpler models due to ability to capture non-linear relationships

## Future Improvements

- Incorporate additional data sources (crime rates, school ratings, etc.)
- Experiment with gradient boosting models (XGBoost, LightGBM)
- Add time-series analysis for price trends
- Implement ensemble methods combining multiple models
- Build interactive dashboard for predictions

## License

This project is for educational purposes.

## Acknowledgments

Based on hands-on machine learning practices and the California Housing dataset from the 1990 census.
