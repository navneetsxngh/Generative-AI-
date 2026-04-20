from fastapi import FastAPI
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from langserve import add_routes

# Load environment variables
load_dotenv()

# Validate API key early
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found in .env")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain"
)

# Optional LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize model 
groq_model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# Prompt Template
system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt_template | groq_model | parser

# Add route
add_routes(
    app,
    chain,
    path="/chain"
)

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)