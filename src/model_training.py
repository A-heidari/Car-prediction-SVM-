from src.feature_engineering import prepare_feature
from src.data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, confusion_matrix)


import matplotlib.pyplot as plt 
import seaborn as sns 

df = load_data("Cars.csv")
df = prepare_feature(df)

x = df.drop(columns=["price"])
y= df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=42,
   
)

numeric_cols = x_train.select_dtypes(
    include=["number"],

).columns.tolist()
print(f"\n ---numerical columns--- ")
print(numeric_cols)

category_cols = x_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
print(f"\n ---categorical columns--- ")
print(category_cols)


# ----------------------------------------------
# numerical scaling 
# ----------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ----------------------------------------------
# categorical scaling 
# ----------------------------------------------

categorical_pipeline = Pipeline(
    steps= [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ----------------------------------------------
# Preprocessor
# ----------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, category_cols)
    ]
)


# ----------------------------------------------
# training model 
# ----------------------------------------------

svr_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("svr", SVR(
            kernel="linear",
            C=1
        ))
    ]
)

svr_model.fit(x_train, y_train)


# ----------------------------------------------
# Evaluate 
# ----------------------------------------------

y_pred = svr_model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"\n --- METRICS ---")
print(f"MAE : {mae}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")


# ----------------------------------
# VISUALIZATION 
# ----------------------------------


plt.figure(figsize=(10,5))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,

)

plt.title("correlation numeric features")
plt.show()