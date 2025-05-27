import os
from dotenv import load_dotenv
from openai import OpenAI
from base64 import b64encode

#set up OpenAI endpoint
def setup_client():
    load_dotenv()
    os.environ['OPENAI_API_KEY']
    open_AI_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=open_AI_api_key,) 

    return client
