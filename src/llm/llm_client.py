import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# def get_llm():
#     return ChatOpenAI(
#         # api_key=os.environ["GROQ_API_KEY"],
#         api_key="",
#         base_url="https://api.groq.com/openai/v1", 
#         # model=os.getenv("llm_model"),
#         model="llama-3.1-8b-instant",
#         temperature=0.1, 
#         max_tokens=8192
#     )


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_tokens=8192,
        google_api_key=os.getenv("GOOGLE_API_KEY2")
    )
