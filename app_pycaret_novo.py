# Imports
import pandas as pd
import streamlit as st
from io import BytesIO
from pycaret.classification import load_model, predict_model

# Função para converter o df para CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para Excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(
        page_title='PyCaret',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write("## Escorando o modelo gerado no PyCaret")
    st.markdown("---")

    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

    # Verifica se há conteúdo carregado na aplicação
    if data_file_1 is not None:
        try:
            if data_file_1.name.endswith(".csv"):
                df_credit = pd.read_csv(data_file_1)
            elif data_file_1.name.endswith(".ftr"):
                df_credit = pd.read_feather(data_file_1)
            else:
                st.error("Formato de arquivo não suportado.")
                st.stop()

            st.success("✅ Arquivo carregado com sucesso!")
            st.dataframe(df_credit.head())

            # Carrega o modelo salvo
            modelo = load_model("modelo_lightgbm_pycaret")

            # Realiza a previsão
            predict = predict_model(modelo, data=df_credit)

            # Mostra os dados escorados
            st.subheader("📊 Dados com previsões")
            st.dataframe(predict.head())

            # Botão de download
            df_xlsx = to_excel(predict)
            st.download_button(
                label='📥 Download',
                data=df_xlsx,
                file_name='previsoes.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        except Exception as e:
            st.error(f"Erro ao processar o arquivo ou aplicar o modelo: {e}")

if __name__ == '__main__':
    main()
