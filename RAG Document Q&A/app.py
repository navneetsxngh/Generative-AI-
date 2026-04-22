import streamlit as st
import os
import time
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
# Load environment variables
load_dotenv()

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# -------------------- ENV SETUP --------------------
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Q&A"

# -------------------- LLM --------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the questions based only on the provided context.\n"
     "Provide the most accurate response.\n"
     "<context>\n{context}\n</context>"),
    ("human", "Question: {input}")
])

# -------------------- VECTOR CREATION --------------------
def create_vector_embeddings():
    if "vectors" not in st.session_state:

        with st.spinner("Creating embeddings..."):

            # Embedding model (FIXED typo)
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load PDFs (make sure /data folder exists)
            loader = PyPDFDirectoryLoader(DATA_PATH)
            docs = loader.load()

            if not docs:
                st.error("❌ No PDF files found in 'data' folder")
                st.stop()

            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_docs = splitter.split_documents(docs)

            # Create FAISS vector store
            st.session_state.vectors = FAISS.from_documents(
                documents=final_docs,
                embedding=st.session_state.embeddings
            )

            st.success("✅ Vector database created!")

# -------------------- STREAMLIT UI --------------------
st.title("📄 RAG Document Q&A")

user_prompt = st.text_input("Ask a question about your documents")

# Button to create embeddings
if st.button("Create Document Embeddings"):
    create_vector_embeddings()

# -------------------- QUERY HANDLING --------------------
if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create embeddings first.")
    else:
        with st.spinner("Thinking..."):

            # Create chains (FIXED variable name)
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()

            response = retrieval_chain.invoke({
                "input": user_prompt
            })

            end = time.process_time()

        # Output answer
        st.subheader("Answer:")
        st.write(response.get("answer", "No answer found"))

        st.caption(f"⏱ Response time: {round(end - start, 2)} sec")

        # Show retrieved chunks
        with st.expander("🔍 Retrieved Context"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")