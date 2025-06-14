import streamlit as st
import pandas as pd
from pycaret.classification import predict_model as predict_classification
from pycaret.regression import predict_model as predict_regression

def make_predictions():
    st.subheader("🔮 Fazer Previsões")
    
    if "model" in st.session_state:
        model = st.session_state["model"]
        model_type = st.session_state["model_type"]
        target_column = st.session_state["target_column"]
        feature_columns = st.session_state["feature_columns"]
        
        st.write(f"Modelo carregado: {model_type}")
        st.write(f"Variável alvo: {target_column}")
        st.write(f"Variáveis de entrada: {feature_columns}")
        
        input_data = {}
        for col in feature_columns:
            input_data[col] = st.number_input(f"Digite o valor para {col}:", value=0.0)
        
        if st.button("Fazer Previsão"):
            try:
                input_df = pd.DataFrame([input_data])
                
                if model_type == "classification":
                    prediction = predict_classification(model, data=input_df)
                    result = prediction["prediction_label"].iloc[0]
                    st.success(f"Previsão: {result}")
                elif model_type == "regression":
                    prediction = predict_regression(model, data=input_df)
                    result = prediction["prediction_label"].iloc[0]
                    st.success(f"Previsão: {result:.2f}")
                    
            except Exception as e:
                st.error(f"Erro na previsão: {str(e)}")
    else:
        st.warning("Nenhum modelo foi treinado ainda. Vá para a seção de Modelagem ML primeiro.")


