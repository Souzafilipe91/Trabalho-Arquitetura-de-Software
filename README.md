# ML Analytics App

## Descrição
Aplicativo de análise de dados com Machine Learning desenvolvido em Python usando Streamlit e PyCaret.

## Funcionalidades
- Upload de qualquer base de dados (CSV, Excel)
- Análise exploratória de dados automatizada
- Seleção de variáveis de entrada e saída
- Modelos de Machine Learning:
  - Classificação
  - Regressão
  - Clusterização
- Análise e avaliação de modelos
- Interface para fazer previsões com novos dados

## Requisitos
- Python 3.11+
- streamlit==1.45.1
- pycaret==3.3.2
- pandas==2.1.4
- numpy==1.26.4
- matplotlib==3.7.5
- seaborn (instalado como dependência)
- plotly==5.24.1
- scikit-learn==1.4.2

## Instalação
```bash
pip install streamlit pycaret pandas numpy matplotlib seaborn plotly scikit-learn
```

## Como executar
```bash
streamlit run ml_analytics_app.py
```

## Uso
1. **Upload de Dados**: Carregue seu arquivo CSV ou Excel, ou use um dos datasets de exemplo
2. **Análise Exploratória**: Visualize estatísticas, distribuições e correlações dos dados
3. **Modelagem ML**: Escolha o tipo de problema (classificação, regressão ou clusterização) e treine modelos
4. **Previsões**: Use o modelo treinado para fazer previsões com novos dados

## Arquitetura
- **Frontend**: Streamlit para interface web interativa
- **Backend**: PyCaret para funcionalidades de Machine Learning
- **Processamento**: Pandas e NumPy para manipulação de dados
- **Visualização**: Matplotlib, Seaborn e Plotly para gráficos

## Autor
Desenvolvido para projeto acadêmico de Machine Learning

