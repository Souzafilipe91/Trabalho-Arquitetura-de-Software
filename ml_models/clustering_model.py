import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.clustering import setup, create_model, assign_model

def run_clustering_model(df, feature_columns, n_clusters):
    if feature_columns:
        data = df[feature_columns].copy()
        data = data.dropna()
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            st.error("Não há variáveis numéricas suficientes para clusterização.")
            return None, None
        
        st.write(f"Dataset preparado: {numeric_data.shape[0]} linhas, {numeric_data.shape[1]} colunas")
        
        with st.spinner("Executando clusterização..."):
            try:
                clu = setup(numeric_data, session_id=123, verbose=False)
                
                kmeans = create_model("kmeans", num_clusters=n_clusters)
                
                predictions = assign_model(kmeans)
                
                st.success("Clusterização executada com sucesso!")
                st.subheader("Resultados da Clusterização")
                st.dataframe(predictions.head())
                
                if len(feature_columns) >= 2:
                    fig = px.scatter(
                        predictions,
                        x=feature_columns[0],
                        y=feature_columns[1],
                        color="Cluster",
                        title="Visualização dos Clusters"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                return kmeans, predictions
            except Exception as e:
                st.error(f"Erro durante a clusterização: {str(e)}")
                return None, None
    return None, None


