from src.feature_engineering import prepare_feature
from src.data_loader import load_data

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------
# Load data
# ----------------------------------------------

df = load_data("Cars.csv")
df = prepare_feature(df)


# ----------------------------------------------
# X / y
# ----------------------------------------------

x = df.drop(columns=["price"])
y = df["price"]


# ----------------------------------------------
# Train / Test split
# ----------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=42,
)


# ----------------------------------------------
# Feature groups
# ----------------------------------------------

numeric_features = [
    "model_year",
    "milage",
    "car_age",
    "milage_per_year",
    "horsepower",
    "engine_liters",
    "cylinders"
]

categorical_features = [
    "brand",
    "fuel_type",
    "transmission",
    "accident"
]


# ----------------------------------------------
# Numerical preprocessing
# ----------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ----------------------------------------------
# Categorical preprocessing
# ----------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ----------------------------------------------
# Preprocessor
# ----------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ----------------------------------------------
# RBF SVR
# ----------------------------------------------

svr_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("svr", SVR(
            kernel="rbf",
            C=1,
            gamma="scale"
        ))
    ]
)


# ----------------------------------------------
# Training
# ----------------------------------------------

svr_model.fit(x_train, y_train)


# ----------------------------------------------
# Prediction
# ----------------------------------------------

y_pred = svr_model.predict(x_test)


# ----------------------------------------------
# Evaluation
# ----------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print(f"MAE : {mae}")
print(f"RMSE: {rmse}")
print(f"R2  : {r2}")