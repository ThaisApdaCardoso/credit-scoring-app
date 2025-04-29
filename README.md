# 📚 Projeto Final — Credit Scoring utilizando PyCaret e Streamlit

## Sobre o Projeto

Este projeto tem como objetivo desenvolver uma solução completa de **Credit Scoring**, simulando um fluxo real de machine learning aplicado ao mercado financeiro. As principais etapas foram:

- **Pré-processamento de dados:** Tratamento de dados faltantes, transformação de variáveis categóricas, padronização e redução de dimensionalidade.
- **Treinamento de modelo supervisionado:** Uso do **PyCaret** para automatizar a comparação de vários modelos de classificação e seleção do melhor modelo.
- **Construção de pipeline de produção:** Criação de um pipeline para processar dados novos e realizar previsões (escoragem).
- **Desenvolvimento de um aplicativo interativo:** Implementação de uma aplicação em **Streamlit** para realizar a escoragem de novos dados.
- **Deploy em ambiente cloud:** Publicação da aplicação no **Render** para acesso via navegador web.

---

## 🚀 Resultados

A aplicação web desenvolvida em Streamlit está disponível no Render:  
🔗 [Acesse o app aqui](https://credit-scoring-app-xrgo.onrender.com/)

Demonstração da aplicação em vídeo:
🎬 https://github.com/user-attachments/assets/17b8664d-3dd0-440c-8b0c-6dbdf40a22b2

---

## 📄 Estrutura do Projeto

- `M38_tarefa1_corrigido.ipynb` → Notebook com o pipeline de pré-processamento, treinamento do modelo de classificação e processo de escoragem.
- `app_pycaret_novo.py` → Código da aplicação web construída em Streamlit.
- `requirements.txt` → Arquivo contendo as dependências necessárias para reprodução do ambiente.
- `modelo_lightgbm_pycaret.pkl` → Arquivo de modelo treinado salvo com o PyCaret.

---

## 🔎 Detalhamento Técnico

- **Pré-processamento:**
  - Imputação de valores ausentes: média para numéricas e moda para categóricas.
  - Normalização utilizando método `zscore`.
  - Redução de dimensionalidade utilizando PCA.
  - Codificação de variáveis categóricas com OneHotEncoder.

- **Modelagem:**
  - Comparação automática de modelos usando PyCaret.
  - Seleção do modelo com melhor desempenho (LightGBM).
  - Avaliação por AUC-ROC, matriz de confusão e feature importance.

- **Aplicativo Streamlit:**
  - Upload de novos datasets em `.csv` ou `.feather`.
  - Aplicação do pipeline de previsão no conjunto carregado.
  - Exibição dos dados escorados na interface.
  - Possibilidade de download dos resultados em formato `.xlsx`.

- **Deploy:**
  - Deploy do app Streamlit no **Render**.
  - Ambiente configurado com `requirements.txt`.

---

## 🛠️ Tecnologias Utilizadas

- Python
- PyCaret
- Streamlit
- Scikit-Learn
- LightGBM
- Pandas
- Render.com (Deploy)

---

## 📥 Como Rodar Localmente

Caso deseje rodar o projeto localmente:

```bash
# Clone o repositório
git clone https://github.com/ThaisApdaCardoso/credit-scoring-app

# Instale as dependências
pip install -r requirements.txt

# Rode o app
streamlit run app_pycaret_novo.py
