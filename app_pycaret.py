# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Configuração inicial da página
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.title("🏦 Credit Scoring com PyCaret")

# Carregar modelo com cache
@st.cache_resource
def carregar_modelo():
    return load_model("modelo_lightgbm_pycaret")

modelo = carregar_modelo()

# Upload de arquivo
uploaded_file = st.file_uploader("📁 Envie o arquivo de dados (.csv ou .ftr)", type=["csv", "ftr"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".ftr"):
            df = pd.read_feather(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado.")
            st.stop()

        st.success("✅ Arquivo carregado com sucesso!")
        st.subheader("📄 Dados carregados:")
        st.dataframe(df.head())

        # 🔮 Aplicar o modelo
        try:
            resultado = predict_model(modelo, data=df)
            st.subheader("📊 Dados com previsões:")
            st.dataframe(resultado.head())

            # ⬇️ Botão de download
            csv = resultado.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar resultado como CSV", data=csv, file_name="previsoes.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Erro ao aplicar o modelo: {e}")

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")

