import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from build_catalog import build_catalog

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

if not os.path.exists("data/coursera_catalog.csv"):
    build_catalog()

catalog = pd.read_csv("data/coursera_catalog.csv")
learners = json.load(open("data/export_learners.json", encoding="utf-8"))

embeddings_path = Path("data/catalog_embeddings.npy")
if os.path.exists(embeddings_path):
    catalog_embeddings = np.load(embeddings_path)
    print("Cache des embeddings chargé depuis data/catalog_embeddings.npy")
else:
    print("Encodage de la colonne combined_text du catalogue...")
    catalog_embeddings = model.encode(catalog["combined_text"].tolist(), show_progress_bar=True)
    np.save(embeddings_path, catalog_embeddings)
    print("Embeddings calculés et sauvegardés dans data/catalog_embeddings.npy")


def build_learner_profile_text(learner):
    parts = [
        learner.get("program_name") or "",
        learner.get("current_project_title") or "",
        learner.get("current_project_description") or "",
        ", ".join(learner.get("validated_projects_titles") or []),
    ]
    return ". ".join(part.strip() for part in parts if part.strip())


def get_top_recommendations(learner, catalog, model, top_n=5):
    profile_text = build_learner_profile_text(learner)
    print("Encodage du profil apprenant...")
    profile_embedding = model.encode([profile_text], show_progress_bar=True)

    scores = cosine_similarity(profile_embedding, catalog_embeddings).flatten()

    catalog_sorted = catalog.assign(score=scores).sort_values("score", ascending=False).head(top_n)
    return catalog_sorted[["title", "Organization", "Level", "URL", "score"]]


if __name__ == "__main__":
    for learner in learners:
        print(f"learner_id: {learner['learner_id']}")
        print(f"program_name: {learner['program_name']}")
        print(f"progress_percentage: {learner['progress_percentage']}")

        recommendations = get_top_recommendations(learner, catalog, model, top_n=3)
        print(recommendations)
        print("-" * 40)