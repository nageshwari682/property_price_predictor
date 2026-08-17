# property Price Prediction

This project builds a machine learning application to predict residential property prices using attributes such as location, area, number of bedrooms, number of bathrooms, age of the property, and other key features.Property Price Prediction is a machine learning project that predicts the estimated price of a property based on various property-related features.
The project demonstrates the complete machine learning workflow, including data generation/collection, data preprocessing, exploratory data analysis, model training, model evaluation, model saving, and price prediction.
## Project Goals
- Clean and preprocess the dataset
- Perform exploratory data analysis (EDA)
- Engineer new predictive features
- Train and compare regression models
- Evaluate and interpret model performance
- Provide a simple prediction interface

## Project Structure

```text
property_price_predictor/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── property_price_eda.ipynb
├── src/
│   ├── property_price_model/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │   ├── model_pipeline.py
│   │   ├── train_model.py
│   │   └── predict.py
│   └── __init__.py
├── reports/
│   └── model_evaluation_report.md
├── models/
│   └── property_price_model.joblib
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python src/property_price_model/train_model.py
```

4. Run the prediction CLI:

```bash
python main.py
```

## Dataset

The project uses a synthetic or real-world residential housing dataset with fields like:
- location
- area_sqft
- bedrooms
- bathrooms
- stories
- age_years
- garage
- balcony
- basement
- nearby_school_score
- crime_rate
- property_type
- target: price

If you want to use a custom dataset, place it in `data/raw/` and update the file path in the training script.

## Model

The project trains a gradient boosting regressor after preprocessing categorical variables and scaling numeric values. A pipeline is used to combine preprocessing and modeling for clean training and deployment.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Jupyter Notebook
- Git
- GitHub
## Evaluation Metrics

The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Example Prediction

```python
from src.property_price_model.predict import predict_property_price

sample = {
    "location": "Downtown",
    "area_sqft": 1850,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "age_years": 12,
    "garage": 1,
    "balcony": 1,
    "basement": 0,
    "nearby_school_score": 8.5,
    "crime_rate": 0.2,
    "property_type": "Apartment"
}

price = predict_property_price(sample)
print(f"Estimated property price: ${price:,.2f}")
```

## Licenses and Notes

This project is intended for educational and portfolio use. You can adapt the code to your own dataset and extend the model with additional features.
