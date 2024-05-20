import asyncio
import threading
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import json

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = [Document(text) for text in loader.load()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    document_chunks = text_splitter.split_documents([doc.page_content for doc in documents])
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return document_chunks, embeddings

class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}

def get_vectorstore_from_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    document = Document(text)
    document_chunks = text_splitter.split_documents([document])
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return document_chunks, embeddings

def get_conversational_chain():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=loop.run_forever).start()
    prompt_template = """
    Create MCQ based on the context:\n {context}?\n
    The MCQ should be sent back in json format. 
    The JSON format should be like:
    Question: ?,
    Options: [opt1, opt2, op3 ...],
    Answer: 

Only send the json back. Do not write anything else. Nothing else other than json. Peroid. Generate maximum amount of MCQs.
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_mcq_chain():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=loop.run_forever).start()
    mcq_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    mcq_prompt = PromptTemplate(template = "{context}\n\nGenerate MCQs in JSON format:", input_variables = ["context"])
    mcq_chain = load_qa_chain(mcq_model, chain_type="stuff", prompt=mcq_prompt)
    return mcq_chain


st.set_page_config(page_title="MCQ Bot", page_icon="🤖")
   
def main():
    st.title("MCQ Bot")

    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if website_url is None or website_url == "" and uploaded_file is None:
        st.info("Please enter a website URL or upload a PDF")
    else:
        if "vector_store" not in st.session_state:
            if website_url is not None and website_url != "":
                st.session_state.vector_store = get_vectorstore_from_url(website_url)
            elif uploaded_file is not None:
                pdf_file = PdfReader(uploaded_file)
                text = ""
                for page in pdf_file.pages:
                    text += page.extract_text()
                st.session_state.vector_store = get_vectorstore_from_text(text)
        # Now that the conditions are met, display the button
        if "vector_store" in st.session_state:
            if st.button("Generate MCQ"):
                document_chunks, embeddings = st.session_state.vector_store
                new_db = FAISS.from_documents(document_chunks, embeddings)
                docs = new_db.similarity_search("Generate MCQ based on the document")
                mcq_chain = get_mcq_chain()
                mcq_response = mcq_chain({"input_documents": docs, "question": "Generate MCQ based on the document"}, return_only_outputs=True)
                
                if mcq_response["output_text"].strip():  # Check if the response is not empty
                    try:
                        # Strip unwanted characters, particularly backticks and language annotations
                        cleaned_response = mcq_response["output_text"].strip().strip("```json").strip("```").strip()
                        
                        # Additional cleaning if necessary
                        cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()

                        # Attempt to load the cleaned response as JSON
                        mcq_json = json.loads(cleaned_response)
                        st.write("MCQs: ", mcq_json)
                    except json.JSONDecodeError as e:
                        st.error(f"The response is not a valid JSON string. Error: {str(e)}")
                        st.error(f"Invalid JSON string: {cleaned_response}")
                else:
                    st.error("The response is empty.")


if __name__ == "__main__":
    main()