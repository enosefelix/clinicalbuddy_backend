import os
import io
import json
import cohere
import threading
import requests
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain_cohere import ChatCohere
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from config.constants import DOMAINS, MED_PROMPTS
from langchain.adapters.openai import convert_openai_messages


load_dotenv()

# Api Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
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


# Helper functions
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def extract_serper_search_website_name_and_url(data):
    extracted_data = []

    for item in data.get("organic", []):
        parsed_url = urlparse(item["link"])
        domain_name = parsed_url.netloc.replace("www.", "")
        extracted_data.append({"website_name": domain_name, "url": item["link"]})

    return extracted_data


def serper_search(final_question):
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps(
            {
                "q": final_question,
                "gl": "gb",
                "type": "search",
                "location": "United Kingdom",
                "num": 11,
                "engine": "google",
            }
        )
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }

        serper_response = requests.request("POST", url, headers=headers, data=payload)
        serper_data = serper_response.json()
        serper_array = serper_data["organic"]
        references = extract_serper_search_website_name_and_url(serper_data)

        prompt = [
            {
                "role": "system",
                "content": f"You are a helpful AI research assistant with specialized expertise in medical sciences."
                f"Your primary function is to synthesize well-structured, critically analyzed, and medically accurate reports based on provided information."
                f"Your responses should emulate the communication style of a medical professional, incorporating appropriate medical terminology and considerations, and always adhere to the present simple tense for consistency",
            },
            {
                "role": "user",
                "content": f'Information: """{serper_array}"""\n\n'
                f"Using the above information, answer the following"
                f'query: "{final_question}" providing very extensive responses'
                f"Ensure your response is structured with medical precision, using markdown syntax for clarity and professionalism. Never include references in your response",
            },
        ]

        lc_messages = convert_openai_messages(prompt)
        answer = openAIChatClient.invoke(lc_messages).content
        response = {
            "answer": answer,
            "references": references,
            "source": "web_search",
            "status": 200,
        }

        return response

    except Exception as e:
        return {
            "answer": "",
            "references": [],
            "source": "web_search",
            "status": 400,
        }


def extract_tavily_search_website_name_and_url(data):
    extracted_data = []

    for item in data:
        parsed_url = urlparse(item["url"])
        domain_name = parsed_url.netloc.replace("www.", "")
        extracted_data.append({"website_name": domain_name, "url": item["url"]})

    return extracted_data


def tavily_search(final_question):
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        tavily_response = client.search(
            query=final_question,
            search_depth="advanced",
            # include_domains=DOMAINS,
            max_results=10,
        )["results"]

        references = extract_tavily_search_website_name_and_url(tavily_response)
        prompt = [
            {
                "role": "system",
                "content": f"You are a helful AI research assistant with specialized expertise in medical sciences."
                f"Your primary function is to synthesize well-structured, critically analyzed, and medically accurate reports based on provided information."
                f"Your responses should emulate the communication style of a medical professional, incorporating appropriate medical terminology and considerations, and always adhere to the present simple tense for consistency",
            },
            {
                "role": "user",
                "content": f'Information: """{tavily_response}"""\n\n'
                f"Using the above information, answer the following"
                f'query: "{final_question}" providing very extensive responses'
                f"Ensure your response is structured with medical precision, using markdown syntax for clarity and professionalism. Never include references in your response",
            },
        ]

        lc_messages = convert_openai_messages(prompt)
        answer = cohereChatClient.invoke(lc_messages).content
        response = {"answer": answer, "references": references}

        return response
    except Exception as e:
        return f"Error occurred: please try again later"


def grade_docs_with_cohere(prompt_rag):
    response = co.chat(message=prompt_rag, model="command-r", temperature=0.0)
    return response.text


def grade_docs_with_openai(prompt_rag):
    response = openAIClient.chat.completions.create(
        messages=[
            {"role": "system", "content": " You are a helpful ai assistant"},
            {"role": "user", "content": prompt_rag},
        ],
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0,
        seed=123,
    )
    return response.choices[0].message.content


def generate_final_response_with_cohere(
    filtered_relevant_ranked_data,
    user_question,
):

    response_prompt = f"""
    ##Context
    Below is relevant context: {filtered_relevant_ranked_data}

    ## User Question
    Here is the user's question: {user_question} \n

    ## Instructions
    You are a helpful AI assistant.
    BASED ON THE PROVIDED CONTEXT, answer the user's question with detailed explanations, listing and highlighting answers where appropriate for enhanced readability. ALWAYS use MLA format and Markdown for clarity and organization, ensuring your answers are thorough and reflect medical expertise. Adhere to the present simple tense for consistency and ensure your answers are ALWAYS grounded in the context and relevant to the question. If the materials are not relevant or complete enough to confidently answer the user's questions, your best response is 'the materials do not appear to be sufficient to provide a good answer'."
    """
    response = co.chat(message=response_prompt, model="command-r", temperature=0.0)
    return response.text


def generate_final_response_with_openai(
    filtered_relevant_ranked_data,
    user_question,
):
    response_prompt = f"""Use only the context below to answer the subsequent question.
    Context:
    \"\"\"
    {filtered_relevant_ranked_data}
    \"\"\"
    Question: {user_question}"""

    response = openAIClient.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": " Utilize use MLA format and Markdown for clarity and organization, ensuring your answers are thorough and reflect medical expertise. Adhere to the present simple tense for consistency. Answer the question with detailed explanations, listing and highlighting answers where appropriate for enhanced readability. Never add reference.",
            },
            {"role": "user", "content": response_prompt},
        ],
        model="gpt-4o",
        temperature=0,
        seed=123,
    )

    return response.choices[0].message.content


def transcribe_audio(file_bytes, file_type, content_type):
    system_prompt = "You are a helpful AI assistant that helps users search through clinical and medical guidelines. Accept the user question, correct any typographical errors and return the users exact words, Do not answer the questions, just return the exact question"

    file_buffer = io.BytesIO(file_bytes)
    file_info = ("temp." + file_type, file_buffer, content_type)
    transcript = openAIClient.audio.translations.create(
        model="whisper-1", file=file_info, response_format="text", prompt=MED_PROMPTS
    )

    corrected_transcript = openAIClient.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )

    return corrected_transcript.choices[0].message.content


def run_with_timeout(func, args=(), timeout=7):
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        # If the thread is still alive after the timeout, it means it's taking too long
        raise TimeoutError
    if exception[0] is not None:
        # If an exception occurred in the thread, raise it
        raise exception[0]
    return result[0]
