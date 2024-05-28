import os
import cohere
from langchain_openai import ChatOpenAI
import instructor
from openai import OpenAI
from dotenv import load_dotenv
from langchain_cohere import ChatCohere


load_dotenv()

# Api Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# LLM clients
co = cohere.Client(COHERE_API_KEY)
openAIClient = OpenAI()
cohereChatClient = ChatCohere(model="command-r")
openAIChatClient = ChatOpenAI(
    temperature=0.0,
    # model="gpt-3.5-turbo-0125",
    model="gpt-4o",
)

instructor_client = instructor.from_openai(openAIClient)