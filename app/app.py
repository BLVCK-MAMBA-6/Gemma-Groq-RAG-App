import os
import streamlit as st
from dotenv import load_dotenv
import time
import asyncio
import nest_asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_groq.chat_models import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain

# Fix for async event loop issues in Streamlit
nest_asyncio.apply()

load_dotenv()

# Load the GROQ and Google API key from the .env File
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

Context: {context}

Question: {input}

Answer:"""
)

def vector_embedding():  # Fixed function name
    if "vectors" not in st.session_state:
        with st.spinner("Creating vector store..."):
            # Use HuggingFace embeddings instead of Google's to avoid async issues
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            st.session_state.loader = PyPDFDirectoryLoader("./pdf")
            st.session_state.documents = st.session_state.loader.load()
            
            if not st.session_state.documents:
                st.error("No PDF documents found in the './pdf' directory. Please add some PDF files.")
                return
                
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("Enter your question on the document here:")

if st.button("Creating Vector Store"):  # Fixed button text
    vector_embedding()  # Fixed function call
    st.write("Vector Store Created Successfully")

# Only process if there's a question AND vectors exist
if prompt1:
    if "vectors" in st.session_state:
        try:
            # Create chains and process query
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            end = time.process_time()
            
            st.write("**Answer:**")
            st.write(response['answer'])
            
            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.write("--------------------")
                    
                    # Check if metadata exists before accessing
                    if 'page' in doc.metadata:
                        st.write(f"Page Number: {doc.metadata['page']}")
                    if 'source' in doc.metadata:
                        st.write(f"Document Name: {doc.metadata['source']}")
                    st.write("--------------------")
                
                st.write(f"Time taken to process the request: {end - start:.4f} seconds")
                st.write("Total Chunks Retrieved: ", len(response["context"]))
                
                # Note: usage and cost info may not be available depending on your setup
                if "usage" in response:
                    st.write("Total Tokens Used: ", response["usage"]["total_tokens"])
                    st.write("Total Cost: $0.0000000000")
                    
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            st.error("Please make sure you have created the vector store first.")
    else:
        st.warning("Please create the vector store first by clicking the 'Creating Vector Store' button.")
