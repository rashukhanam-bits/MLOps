import pandas as pd
from ucimlrepo import fetch_ucirepo

def download():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    df.to_csv("data/raw/heart.csv", index=False)
    print(f"Saved {df.shape[0]} rows to data/raw/heart.csv")

if __name__ == "__main__":
    download()