import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

import os
from dotenv import load_dotenv

 #── TOOLS SETUP ───────────────────────────────────────────────────────────────
# Each tool lets the agent search a different source
 
# Arxiv: searches research papers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
 
# Wikipedia: searches Wikipedia articles
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
 
# DuckDuckGo: general web search
search = DuckDuckGoSearchRun(name="Search")
 
tools = [search, arxiv, wiki]
 
# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.title("🔎 LangChain - Chat with Search")
st.caption("Ask me anything — I'll search the web, Wikipedia, or Arxiv to answer!")
 
# Sidebar: user enters their Groq API key securely
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
 
# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
# session_state persists data across Streamlit reruns (like a global variable)
# BUG 1 FIX: "assisstant" → "assistant" (typo caused role icon to not render)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot who can search the web. How can I help you?"}
    ]
 
# Display all previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
 
# ── CHAT INPUT & AGENT RESPONSE ───────────────────────────────────────────────
if prompt := st.chat_input(placeholder="What is machine learning?"):
 
    # Save and display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
 
    # Only proceed if an API key has been entered
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()
 
    # Initialize the LLM (Groq's fast Llama model with streaming enabled)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", streaming=True)
 
    # ZERO_SHOT_REACT_DESCRIPTION: agent decides which tool to use based on the question
    # BUG 2 FIX: "handling_parsing_errors" → "handle_parsing_errors" (wrong keyword caused TypeError)
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
 
    with st.chat_message("assistant"):
        # StreamlitCallbackHandler shows the agent's reasoning steps live in the UI
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
 
        # BUG 3 FIX: pass `prompt` (a string) not `st.session_state.messages` (a list of dicts)
        # agent.run() expects a plain string — passing a list caused a serialization error
        response = search_agent.run(prompt, callbacks=[st_cb])
 
        # Save and display the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
 