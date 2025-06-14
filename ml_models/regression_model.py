import streamlit as st
from pycaret.regression import setup, compare_models, finalize_model, pull

def train_regression_model(data, target_column):
    st.write(f"Dataset preparado: {data.shape[0]} linhas, {data.shape[1]} colunas")
    with st.spinner("Treinando modelo de regressão..."):
        try:
            reg = setup(data, target=target_column,
                        session_id=123, train_size=0.8, verbose=False)

            best_models = compare_models(
                include=["lr", "rf", "dt", "gbr", "knn"],
                sort="MAE",
                n_select=3,
                verbose=False
            )

            best_model = best_models[0]
            final_model = finalize_model(best_model)

            st.success("Modelo de regressão treinado com sucesso!")
            st.subheader("Métricas do Modelo")
            results = pull()
            st.dataframe(results)
            return final_model
        except Exception as e:
            st.error(f"Erro durante o treinamento: {str(e)}")
            return None


