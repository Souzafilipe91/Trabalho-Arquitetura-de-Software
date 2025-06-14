import streamlit as st
import pandas as pd
from pycaret.datasets import get_data

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            st.error("Formato de arquivo não suportado. Use CSV ou Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {str(e)}")
        return None

def load_example_data(selected_example):
    try:
        example_datasets = {
            "Boston Housing (Regressão)": "boston",
            "Titanic (Classificação)": "titanic",
            "Diabetes (Regressão)": "diabetes",
            "Wine (Classificação)": "wine"
        }
        df = get_data(example_datasets[selected_example])
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dataset: {str(e)}")
        return None


