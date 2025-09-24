import os
from dotenv import load_dotenv

def load_env():
    """Load OpenAI API key from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key
