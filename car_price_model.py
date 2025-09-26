# car_price_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Data cleaning functions
def clean_price(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "Ask" in x:
        return np.nan
    try:
        return int(x.replace(",", "").strip())
    except:
        return np.nan

def clean_kms_driven(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "kms" in x:
        try:
            return int(x.replace(",", '').replace('kms', '').strip())
        except:
            return np.nan
    else:
        return np.nan

def load_and_preprocess_data():
    """Load and preprocess the data for training"""
    df = pd.read_csv("quikr_car.csv")
    
    # Apply cleaning
    df["Price"] = df["Price"].apply(clean_price)
    df["kms_driven"] = df['kms_driven'].apply(clean_kms_driven)
    
    # Handle missing values
    df['fuel_type'].fillna(df['fuel_type'].mode()[0], inplace=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    
    # Remove rows with missing values
    df = df.dropna(subset=["Price", "year"]).reset_index(drop=True)
    df = df.drop(columns=["name"])
    
    # Feature engineering
    current_year = 2024
    df['car_age'] = current_year - df['year']
    df = df.drop(columns=['year'])
    
    # Remove outliers
    Q1 = df['Price'].quantile(0.05)
    Q3 = df['Price'].quantile(0.95)
    IQR = Q3 - Q1
    df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)]
    
    return df

def train_and_save_model():
    """Train the model and save it"""
    df = load_and_preprocess_data()
    
    # Prepare features and target
    X = df.drop(columns=["Price"])
    y = df["Price"]
    
    # Define feature types
    cat_features = ["company", "fuel_type"]
    num_features = ["car_age", "kms_driven"]
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )
    
    # Create and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'car_price_model.pkl')
    print("Model trained and saved successfully!")
    
    return model

# Train the model (run this once)
if __name__ == "__main__":
    train_and_save_model()