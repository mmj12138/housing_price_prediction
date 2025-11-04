import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import CONFIG, TRAIN_PATH, TEST_PATH

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f'✅ Loaded Train: {train_df.shape}, Test: {test_df.shape}')
    return train_df, test_df

def preprocess_data(train_df):
    target = CONFIG['target']

    # === Feature groups (structural chosen for initialization) ===
    structural_features = [
        'LotArea','OverallQual','YearBuilt','YearRemodAdd','GrLivArea',
        'FullBath','BedroomAbvGr','TotRmsAbvGrd','GarageCars','GarageArea',
        'TotalBsmtSF','1stFlrSF','BsmtFinSF1','Fireplaces'
    ]
    socioeconomic_features = []  # reserved for later

    features = structural_features + socioeconomic_features

    # Work on a copy with only selected features + target
    df = train_df[features + [target]].copy()

    # Simple missing value handling: numeric -> median
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('Missing')

    X = df[features]
    y = df[target]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # split train -> train/val (from Kaggle train.csv)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG['val_size'], random_state=CONFIG['random_state']
    )

    return preprocessor, X_train, X_val, y_train, y_val, features
