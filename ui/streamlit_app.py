import os
import requests
import streamlit as st


# Базовый URL FastAPI сервиса. Можно переопределить через переменную окружения.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")
st.title("📧 Spam Email Classifier")
st.caption("UI для сервиса классификации писем (FastAPI + DistilBERT)")

st.write(f"Текущий API_URL: `{API_URL}`")

text = st.text_area("Текст письма", height=200, placeholder="Вставь сюда текст письма...")

col1, col2 = st.columns(2)
with col1:
    do_predict = st.button("Проверить", type="primary")
with col2:
    st.button("Очистить", on_click=lambda: st.session_state.update({"_clear": True}))

if do_predict:
    if not text.strip():
        st.warning("Введи текст письма.")
    else:
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=30,
            )
            if resp.status_code != 200:
                st.error(f"Ошибка API: {resp.status_code}\n{resp.text}")
            else:
                data = resp.json()
                label = data.get("label")
                score = data.get("score")

                st.subheader("Результат")
                st.metric("Класс", label)
                st.metric("Score", f"{score:.4f}" if isinstance(score, (int, float)) else str(score))

        except requests.exceptions.RequestException as e:
            st.error(f"Не удалось обратиться к API: {e}")
