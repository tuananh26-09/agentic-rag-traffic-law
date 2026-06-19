# import os
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# def get_llm():
#     return ChatOpenAI(
#         api_key=os.environ["GROQ_API_KEY"],
#         base_url="https://api.groq.com/openai/v1", 
#         model=os.getenv("llm_model"),
#         temperature=0.0, 
#         max_tokens=8192
#     )


import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )