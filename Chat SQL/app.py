import streamlit as st
from pathlib import Path
from urllib.parse import quote_plus

# LangChain imports
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

from sqlalchemy import create_engine
import sqlite3

# ---------------- UI ---------------- #
st.set_page_config(page_title="Chat with SQL DB", page_icon="🧠")
st.title("🧠 Chat with SQL Database")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite (student.db)", "Connect to MySQL"]
selected_opt = st.sidebar.radio("Choose Database", radio_opt)

# ---------------- DB CONFIG ---------------- #
if selected_opt == "Connect to MySQL":
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
    mysql_port = st.sidebar.text_input("MySQL Port", value="3306")
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("Database Name", value="studentdb")
else:
    db_uri = LOCALDB

# API Key
api_key = st.sidebar.text_input("Groq API Key", type="password")

if not api_key:
    st.info("Please add your Groq API Key")
    st.stop()

# ---------------- LLM ---------------- #
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    streaming=True
)

# ---------------- DATABASE FUNCTION ---------------- #
@st.cache_resource(ttl="2h")
def configure_db():
    if db_uri == LOCALDB:
        db_path = (Path(__file__).parent / "student.db").absolute()
        return SQLDatabase(create_engine(f"sqlite:///{db_path}"))

    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please fill all MySQL fields")
            st.stop()

        encoded_password = quote_plus(mysql_password)

        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{encoded_password}@{mysql_host}:{mysql_port}/{mysql_db}"
        )
        return SQLDatabase(engine)

# Create DB
db = configure_db()

# ---------------- AGENT ---------------- #
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# ---------------- CHAT MEMORY ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your database 👀"}
    ]

# Display messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- USER INPUT ---------------- #
user_query = st.chat_input("Ask a question about your DB")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())

        response = agent.invoke(
            {"input": user_query},
            config={"callbacks": [callback]}
        )

        output = response["output"] if isinstance(response, dict) else response

        st.session_state.messages.append({"role": "assistant", "content": output})
        st.write(output)