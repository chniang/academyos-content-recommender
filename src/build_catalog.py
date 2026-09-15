import pandas as pd
from datasets import load_dataset

dataset = load_dataset("azrai99/coursera-course-dataset", split="train")

df = dataset.to_pandas()
df = df[["title", "URL", "Description", "Skills", "Level", "Organization"]]
df = df.dropna(subset=["Description"])
df = df[df["Description"].str.strip() != ""]
df["Level"] = df["Level"].fillna("Non précisé")

df["combined_text"] = (
    df["title"].astype(str)
    + ". "
    + df["Description"].astype(str)
    + ". "
    + df["Skills"].astype(str)
)

df.to_csv("data/coursera_catalog.csv", index=False)
print(f"Nombre de lignes gardées : {len(df)}")
