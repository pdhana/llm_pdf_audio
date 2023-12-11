import streamlit as st
import pickle
import os

from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import whisper, openai


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM ChatBot Dashboard ')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                
      Alternatively this can be done using Drag & Drop UI which requires any of these Cloud Environment like
      Azure, Google Cloud or AWS. Also we can use open source frame works like pinecone, vectra etc but 
      free version has very limited functionality. 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ for Extentia')

load_dotenv()


def audio_file(audio):
    """ limited 200MB, you could increase by `streamlit run foo.py --server.maxUploadSize=1024` """

    if audio is not None:
        result = openai.Audio.transcribe("whisper-1", audio, verbose=True)
        st.write(result["text"])

def pdf_file(pdf):
        # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
                st.write( cb ) 
            st.write(response)

def main():
    st.header("A.B.A Admin Dashboard 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf' )
    pdf_file( pdf)

    # upload a Audio file
    audio = st.file_uploader("Upload an audio file", type=["mp3"])
    audio_file(audio)


if __name__ == '__main__':
    main()


