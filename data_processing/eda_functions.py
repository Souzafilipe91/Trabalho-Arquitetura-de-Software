import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_analysis(df):
    st.subheader("📊 Análise Exploratória de Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Linhas", df.shape[0])
    with col2:
        st.metric("Colunas", df.shape[1])
    with col3:
        st.metric("Valores Nulos", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicatas", df.duplicated().sum())
    
    st.subheader("Primeiras 10 linhas")
    st.dataframe(df.head(10))
    
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe())
    
    st.subheader("Tipos de Dados")
    types_df = pd.DataFrame({
        'Coluna': df.columns,
        'Tipo': df.dtypes,
        'Valores Únicos': [df[col].nunique() for col in df.columns],
        'Valores Nulos': [df[col].isnull().sum() for col in df.columns]
    })
    st.dataframe(types_df)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) > 0:
        st.subheader("Distribuições das Variáveis Numéricas")
        
        fig, axes = plt.subplots(min(len(numeric_columns), 4), 1, figsize=(10, 3*min(len(numeric_columns), 4)))
        if len(numeric_columns) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_columns[:4]):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
                axes[i].set_title(f'Distribuição de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequência')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if len(numeric_columns) > 1:
            st.subheader("Matriz de Correlação")
            correlation_matrix = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)


