import streamlit as st
import streamlit_chat as st_message
from rag import QAModel as RAGModel
import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv()

st.set_page_config('Servicio Cliente')
GOOGLE_API_KEY = st.text_input('GOOGLE_API_KEY', type='password')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


@st.cache_resource()
def init_qamodel():
    return RAGModel()


if __name__ == "__main__":
    app = QAApp()
    app.run_app()







