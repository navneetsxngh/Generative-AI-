import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

## LangSmith Tracking
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Groq"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),  
        ("user", "Question:{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    llm = ChatGroq(
        model=llm,
        temperature=temperature,        
        max_tokens=max_tokens           
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

## Title
st.title("Enhanced Q&A Chatbot With Groq")  

## Sidebar
st.sidebar.title("Settings")  

## Select Groq model
llm = st.sidebar.selectbox(
    "Select Groq model",
    ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "qwen/qwen3-32b", "whisper-large-v3-turbo"]  
)

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=2000, value=150)

## Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

# ✅ removed stray backticks
if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide a question above")