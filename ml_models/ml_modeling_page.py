import streamlit as st
import pandas as pd
from ml_models.classification_model import train_classification_model
from ml_models.regression_model import train_regression_model
from ml_models.clustering_model import run_clustering_model

def ml_modeling_page(df):
    st.subheader("🤖 Modelagem de Machine Learning")
    
    problem_type = st.selectbox(
        "Selecione o tipo de problema:",
        ["Classificação", "Regressão", "Clusterização"]
    )
    
    columns = df.columns.tolist()
    
    if problem_type in ["Classificação", "Regressão"]:
        target_column = st.selectbox("Selecione a variável alvo:", columns)
        feature_columns = st.multiselect(
            "Selecione as variáveis de entrada:",
            [col for col in columns if col != target_column],
            default=[col for col in columns if col != target_column][:5]
        )
        
        if target_column and feature_columns:
            data = df[[target_column] + feature_columns].copy()
            data = data.dropna()
            
            if len(data) == 0:
                st.error("Não há dados suficientes após remover valores nulos.")
                return
            
            if st.button("Treinar Modelo"):
                if problem_type == "Classificação":
                    final_model = train_classification_model(data, target_column)
                    if final_model:
                        st.session_state["model"] = final_model
                        st.session_state["model_type"] = "classification"
                        st.session_state["target_column"] = target_column
                        st.session_state["feature_columns"] = feature_columns
                elif problem_type == "Regressão":
                    final_model = train_regression_model(data, target_column)
                    if final_model:
                        st.session_state["model"] = final_model
                        st.session_state["model_type"] = "regression"
                        st.session_state["target_column"] = target_column
                        st.session_state["feature_columns"] = feature_columns
    
    elif problem_type == "Clusterização":
        feature_columns = st.multiselect(
            "Selecione as variáveis para clusterização:",
            columns,
            default=columns[:5] if len(columns) >= 5 else columns
        )
        
        n_clusters = st.slider("Número de clusters:", 2, 10, 3)
        
        if feature_columns:
            kmeans_model, predictions = run_clustering_model(df, feature_columns, n_clusters)
            if kmeans_model and predictions is not None:
                st.session_state["cluster_model"] = kmeans_model
                st.session_state["cluster_data"] = predictions
                st.session_state["feature_columns"] = feature_columns


