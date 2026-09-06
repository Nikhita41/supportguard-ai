import pandas as pd

df = pd.read_csv("../data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nUnique intents:", df["intent"].nunique())
print("Unique categories:", df["category"].nunique())

print("\nIntent distribution:")
print(df["intent"].value_counts())