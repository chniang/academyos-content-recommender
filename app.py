import streamlit as st

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from recommender import learners, catalog, model, get_top_recommendations

st.set_page_config(page_title="AcademyOS Content Recommender", layout="wide")

st.title("AcademyOS Content Recommender")

st.markdown(
    "Prototype de recommandation de contenu pédagogique par embeddings pour les apprenants AcademyOS."
)

learner_ids = [learner["learner_id"] for learner in learners]
labels = [f"{learner['learner_id']} ({learner.get('program_name')})" for learner in learners]

selected_label = st.selectbox("Sélectionner un apprenant", labels)
selected_index = labels.index(selected_label)
learner = learners[selected_index]

col1, col2, col3 = st.columns(3)
col1.metric("Programme", learner.get("program_name"))
col2.metric("Progression (%)", learner.get("progress_percentage"))
col3.metric("Projet actuel", learner.get("current_project_title"))

top_n = st.slider("Nombre de recommandations", min_value=3, max_value=10, value=5)

if st.button("Obtenir des recommandations"):
    with st.spinner("Calcul des recommandations en cours..."):
        recommendations = get_top_recommendations(learner, catalog, model, top_n=top_n)

    for _, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"**{row['title']}**")
            st.markdown(f"Organisation : {row['Organization']} | Niveau : {row['Level']}")
            st.markdown(f"Score de similarité : {row['score'] * 100:.2f}%")
            st.markdown(f"[Voir le cours sur Coursera]({row['URL']})")
            st.markdown("---")