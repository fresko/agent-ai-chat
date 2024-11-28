import streamlit as st
import streamlit_chat as st_message
from src.rag import QAModel as RAGModel
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

class QAApp:
    def __init__(self):
        self.qamodel = init_qamodel()
        
    def generate_response(self):
        request = st.session_state.request
        response = self.qamodel(request=request)

        st.session_state.history.append({"message": request,"is_user":True})
        st.session_state.history.append({"message": response['answer'],"is_user":False})

        st.session_state.something = st.session_state.request
        st.session_state.input_text = ""

    def run_app(self):
        st.title("Agente Servicio al Cliente")
        if "history" not in st.session_state:
            st.session_state.history = []
        if st.button("Limpiar Chat"):
            st.session_state.history = []
             
        st.text_input("Escribe tu mensaje", key="request", on_change=self.generate_response)
        #st_message.chat_input("input_text", placeholder="Escribe tu mensaje")

        for i,chat in enumerate(st.session_state.history):
            st.message(**chat,key=str(i))

if __name__ == "__main__":
    app = QAApp()
    app.run_app()







