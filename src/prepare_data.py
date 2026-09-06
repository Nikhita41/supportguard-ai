from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("../data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
# First split: 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["intent"],
    random_state=42
)

# Second split: temporary → 10% validation + 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["intent"],
    random_state=42
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)