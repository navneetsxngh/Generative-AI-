import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
# from langchain_community.llms import ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")   ## --> LangSmith Tracking
os.environ['LANGCHAIN_TRACING_V2'] = "true"                        ## ---> LangSmith Tracing
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")


prompt = ChatPromptTemplate([
    ("system", "You are a helpful assisstant. Please respond to the questions asked"),
    ("user", "Question: {question}")
])

## Streamlit Framework
st.title("Simple Langchain APP using Google Gemini")
input_text = st.text_input("What Question you have in Mind?")

# llm = ollama("gemma:2b")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
chain = prompt | llm | StrOutputParser()

if input_text:
    st.write(chain.invoke({"question" : input_text}))