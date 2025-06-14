import streamlit as st
from ml_analytics_app_refactored.data_processing.data_loader import load_data, load_example_data
from ml_analytics_app_refactored.data_processing.eda_functions import exploratory_analysis
from ml_analytics_app_refactored.ml_models.ml_modeling_page import ml_modeling_page
from ml_analytics_app_refactored.ml_models.prediction_functions import make_predictions

import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="ML Analytics App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 ML Analytics App")
st.markdown("### Aplicativo de Análise de Dados com Machine Learning")
st.markdown("---")

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.selectbox(
    "Escolha uma opção:",
    ["Upload de Dados", "Análise Exploratória", "Modelagem ML", "Previsões"]
)

# Navegação principal
if page == "Upload de Dados":
    st.subheader("📁 Upload de Dados")
    
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV ou Excel",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['data'] = df
            st.success(f"Arquivo carregado com sucesso! {df.shape[0]} linhas e {df.shape[1]} colunas.")
            st.dataframe(df.head())
    
    st.subheader("📊 Ou use dados de exemplo")
    example_datasets_keys = [
        "Boston Housing (Regressão)",
        "Titanic (Classificação)",
        "Diabetes (Regressão)",
        "Wine (Classificação)"
    ]
    selected_example = st.selectbox("Escolha um dataset de exemplo:", example_datasets_keys)
    
    if st.button("Carregar Dataset de Exemplo"):
        df = load_example_data(selected_example)
        if df is not None:
            st.session_state['data'] = df
            st.success(f"Dataset {selected_example} carregado com sucesso!")
            st.dataframe(df.head())

elif page == "Análise Exploratória":
    if 'data' in st.session_state:
        exploratory_analysis(st.session_state['data'])
    else:
        st.warning("Por favor, carregue um dataset primeiro na seção 'Upload de Dados'.")

elif page == "Modelagem ML":
    if 'data' in st.session_state:
        ml_modeling_page(st.session_state['data'])
    else:
        st.warning("Por favor, carregue um dataset primeiro na seção 'Upload de Dados'.")

elif page == "Previsões":
    make_predictions()

# Footer
st.markdown("---")
st.markdown("**ML Analytics App** - Desenvolvido com Streamlit e PyCaret")


