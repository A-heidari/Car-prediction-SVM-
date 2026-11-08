from src.data_loader import load_data

import matplotlib.pyplot as plt 
import seaborn as sns

df = load_data("Cars.csv")
print(df.head())

# ----------------------------------
# Basic information
# ----------------------------------

print("Shape: ")
print(df.shape)

print("\nColumns: ")
print(df.columns)

print("\nInformation: ")
df.info()
# ----------------------------------
# Missing values
# ----------------------------------

print("\nMissing values: ")
print(df.isnull().sum())

# ----------------------------------
# Duplicated
# ----------------------------------

print("\nDuplicates: ")
print(df.duplicated().sum())

# ----------------------------------
# Statistic
# ----------------------------------

print("\nStatistic: ")
print(df.describe())

# ----------------------------------
# Data type
# ----------------------------------

print("\nData type: ")
print(df.dtypes)

# ----------------------------------
# Unique
# ----------------------------------

print("\nUnique values: ")
for column in df.columns:
    print(f"{column}: {df[column].nunique()}")


# ----------------------------------
# milage / price format 
# ----------------------------------

print("\nSample milage values: ")
print(df["milage"].head(10))

print("\nSample price values: ")
print(df["price"].head(10))


# ----------------------------------
# Checking missing values
# ----------------------------------

print("\nMissing values example: ")

print("\nfuel_type: ")
print(df[df["fuel_type"].isna()].head())

print("\naccident: ")
print(df[df["accident"].isna()].head())

print("\nClean_title: ")
print(df[df["clean_title"].isna()].head())


# ----------------------------------
# unique values 
# ----------------------------------

print("\nSelected unique values: ")

print("\nFuel types: ")
print(df["fuel_type"].unique())

print("\nAccident: ")
print(df["accident"].unique())

print("\nClean title: ")
print(df["clean_title"].unique())

print("\nEngine samples: ")
print(df["engine"].unique()[:20])

