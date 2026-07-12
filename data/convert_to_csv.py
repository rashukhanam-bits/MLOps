from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "raw" / "processed.cleveland.data"
OUTPUT_FILE = BASE_DIR / "raw" / "heart.csv"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found at: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, header=None, names=columns)

# Convert target to binary
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

df.to_csv(OUTPUT_FILE, index=False)

print(df.head())
print(f"CSV created successfully at {OUTPUT_FILE}")