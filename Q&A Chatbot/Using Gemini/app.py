import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, please respond to the user's query"),
    ("user", "Question: {question}")
])

def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = GoogleGenerativeAI(
        model=engine,
        google_api_key=api_key,      
        temperature=temperature,      
        max_output_tokens=max_tokens  
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

## Title
st.title("Enhanced Q&A Chatbot With Google Gemini")  

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")  

## Select the Gemini model
engine = st.sidebar.selectbox(
    "Select Gemini model",
    ["gemini-1.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"] 
)

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter your Google API Key in the sidebar")  
else:
    st.write("Please provide a question above")