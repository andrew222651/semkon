import openai

from .env_vars import settings


openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
