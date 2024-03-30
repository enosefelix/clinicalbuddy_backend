import os
import io
import uuid
import boto3
import time
from datetime import datetime
import hashlib
from urllib.parse import urlparse
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from flask import jsonify
from qdrant.qdrant import qdrant_vector_embedding
from firestore.firestore import fetch_missing_pdfs_from_firestore
from helpers.constants import UserClusters, MED_PROMPTS, LOCAL_FRONT_END_URL
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.adapters.openai import convert_openai_messages
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableBranch,
    RunnableParallel,
)
from langchain.memory import ChatMessageHistory

import logging
import requests
import json


# Configure logging
logging.basicConfig(filename="conversation.log", level=logging.DEBUG)


load_dotenv()
tavily_store = {}
openAIClient = OpenAI()
SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


s3_client = boto3.client(
    service_name="s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

openAIChatClient = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0125",
)


chat_history = ChatMessageHistory()

domains = [
    "https://cks.nice.org.uk/",
    "https://www.nice.org.uk/guidance",
    "https://dermnetnz.org/",
    "https://www.pcds.org.uk/",
    "https://pubmed.ncbi.nlm.nih.gov/",
    "https://www.cochranelibrary.com/",
    "https://www.fsrh.org/standards-and-guidance/",
    "https://patient.info/patientplus",
    "https://www.pcsg.org.uk/",
    "https://gpnotebook.com/",
    "https://www.sign.ac.uk/our-guidelines/",
    "https://rightdecisions.scot.nhs.uk/scottish-palliative-care-guidelines/",
    "https://www.westmidspallcare.co.uk/",
    "https://rightdecisions.scot.nhs.uk/scottish-palliative-care-guidelines/",
    "https://gpifn.org.uk/",
    "https://www.brit-thoracic.org.uk/quality-improvement/guidelines/",
    "https://labtestsonline.org.uk/",
    "https://litfl.com/",
    "https://www.msdmanuals.com/en-gb/professional",
    "https://www.rightbreathe.com/",
    "https://www.nhs.uk/conditions/",
    "https://www.nhsinform.scot/illnesses-and-conditions",
    "https://www.nhsinform.scot/self-help-guides",
    "https://patient.info/",
]


def get_pdf_data(pdf_docs):
    pdf_data = []

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.filename
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            pdf_data.append(
                {"pdf_name": pdf_name, "pdf_text": page_text, "page_num": page_num + 1}
            )

    return pdf_data


def map_pdf(pdf_data):
    pdf_mapping = []

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ",", "\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    for i in pdf_data:
        pdf_name = i["pdf_name"]
        page_num = i["page_num"]
        pdf_chunk = text_splitter.split_text(i["pdf_text"])
        for i, chunk in enumerate(pdf_chunk):
            pdf_mapping.append(
                {
                    "pdf_id": f"pdf_{str(uuid.uuid4())}",
                    "pdf_name": pdf_name,
                    "page_num": page_num,
                    "pdf_chunk": chunk,
                }
            )
    return pdf_mapping


def delete_pdf_from_s3bucket(pdf_names):
    if not isinstance(pdf_names, list):
        pdf_names = [pdf_names]

    objects_to_delete = [{"Key": key} for key in pdf_names]

    try:
        response = s3_client.delete_objects(
            Bucket=AWS_S3_BUCKET_NAME, Delete={"Objects": objects_to_delete}
        )

        if "Deleted" in response:
            deleted_objects = response["Deleted"]
            deleted_keys = [deleted_object["Key"] for deleted_object in deleted_objects]
            return (jsonify({"status": 200, "deleted_objects": deleted_keys}),)
        else:
            return (jsonify({"status": 400}),)

    except Exception as e:
        return (jsonify({"status": 400}),)


def upload_pdf_to_s3bucket_and_get_info(uploaded_file, cluster, user_name, category):
    try:
        user_folder = f"{user_name}/"
        object_name = user_folder + uploaded_file.filename
        uploaded_content = uploaded_file.read()
        uploaded_file.seek(0)

        response = s3_client.upload_fileobj(
            uploaded_file,
            AWS_S3_BUCKET_NAME,
            object_name,
        )

        # Construct the URL based on the bucket name and object key
        pdf_url = (
            f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_name}"
        )

        # Return information about the uploaded PDF
        return {
            "pdf_name": uploaded_file.filename,
            "hash_value": hashlib.sha256(uploaded_content).hexdigest(),
            "cluster": cluster,
            "user_name": user_name,
            "pdf_url": pdf_url,
            "category": category,
        }

    except Exception as e:
        return jsonify({"status": 500, "message": "Internal Server Error"})


def upload_pdf_to_qdrant(pdf_files, cluster, category, user_name):
    try:
        raw_data = get_pdf_data(pdf_files)
        mapped_pdf = map_pdf(raw_data)
        metadatas = []
        texts_datas = []

        if len(mapped_pdf) > 0:
            for pdf in mapped_pdf:
                metadata = {
                    "group_id": cluster,
                    "user_name": user_name,
                    "pdf_id": pdf["pdf_id"],
                    "source": pdf["pdf_name"],
                    "page_num": pdf["page_num"],
                    "category": category,
                }
                metadatas.append(metadata)
                texts_datas.append(pdf["pdf_chunk"])

            qdrant_vector_embedding.add_texts(texts=texts_datas, metadatas=metadatas)

        else:
            return {"error": str(e)}

        return {"status": "success"}

    except Exception as e:
        return {"error": str(e)}


def open_ai_conversation_without_history(
    user_question, retriever_filter, fetched_missing_pdfs
):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    filter_response = retriever_filter.invoke(user_question)

    pdf_dict = {}

    for doc in filter_response:
        pdf_name = doc.metadata.get("source")
        page_num = doc.metadata.get("page_num")
        if pdf_name not in pdf_dict:
            pdf_url = next(
                (
                    pdf_info["pdf_url"]
                    for pdf_info in fetched_missing_pdfs
                    if pdf_info["pdf_name"] == pdf_name
                ),
                None,
            )
            pdf_dict[pdf_name] = {
                "pdf_name": pdf_name,
                "pdf_url": pdf_url,
                "pages": [],
            }
        pdf_dict[pdf_name]["pages"].append(page_num)

    pdfs_and_pages = list(pdf_dict.values())

    formatted_response = format_docs(filter_response)

    query = f"""Use the context below to answer the subsequent question. If the answer cannot be found, write "I don't know."

    Context:
    \"\"\"
    {formatted_response}
    \"\"\"

    Question: {user_question}"""

    start_time = time.time()
    # print("Start time:", datetime.fromtimestamp(start_time).strftime("%I:%M %p"))

    response = openAIClient.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": " Utilize use MLA format and Markdown for clarity and organization, ensuring your answers are thorough and reflect medical expertise. Adhere to the present simple tense for consistency. Answer the question with detailed explanations, listing and highlighting answers where appropriate for enhanced readability",
            },
            {"role": "user", "content": query},
        ],
        model="gpt-3.5-turbo-1106",
        temperature=0,
        seed=123,
    )

    final_response = response.choices[0].message.content
    # print("Filter response:", final_response)

    # Stop the timer
    end_time = time.time()
    # print("End time:", datetime.fromtimestamp(end_time).strftime("%I:%M %p"))
    # Calculate the elapsed time

    elapsed_time = end_time - start_time
    # print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    return {"answer": final_response, "pdfs_and_pages": pdfs_and_pages, "status": 200}


def langchain_conversation_without_history(
    user_question, retriever_filter, fetched_missing_pdfs
):
    template = """Answer the user's questions based on the below context. If the context lacks sufficient information, respond with 'I don't know' rather than speculating. Utilize use MLA format and Markdown for clarity and organization, ensuring your answers are thorough and reflect medical expertise. Adhere to the present simple tense for consistency. Answer the question with detailed explanations, listing and highlighting answers where appropriate for enhanced readability. {context} 
    Question: {question}"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | openAIChatClient
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever_filter, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    start_time = time.time()
    # print("Start time:", datetime.fromtimestamp(start_time).strftime("%I:%M %p"))

    response = rag_chain_with_source.invoke(user_question)

    pdf_dict = {}

    for doc in response["context"]:
        pdf_name = doc.metadata.get("source")
        page_num = doc.metadata.get("page_num")
        if pdf_name not in pdf_dict:
            pdf_url = next(
                (
                    pdf_info["pdf_url"]
                    for pdf_info in fetched_missing_pdfs
                    if pdf_info["pdf_name"] == pdf_name
                ),
                None,
            )
            pdf_dict[pdf_name] = {
                "pdf_name": pdf_name,
                "pdf_url": pdf_url,
                "pages": [],
            }
        pdf_dict[pdf_name]["pages"].append(page_num)

    pdfs_and_pages = list(pdf_dict.values())
    answer = response["answer"]

    final_response = answer

    # print("final response:", final_response)
    # Stop the timer
    end_time = time.time()
    # print("End time:", datetime.fromtimestamp(end_time).strftime("%I:%M %p"))

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    return {"answer": answer, "pdfs_and_pages": pdfs_and_pages, "status": 200}


def langchain_conversation_with_history(
    user_question, session_id, chat_history, retriever_filter, fetched_missing_pdfs
):
    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context. If the context doesn't contain any relevant information to the question, don't make something up and just say 'I don't know'. Utilize  Markdown for clarity and organization, ensuring your answers are thorough and reflect medical expertise. Adhere to the present simple tense for consistency. Answer the question with detailed explanations, listing and highlighting answers where appropriate for enhanced readability: {context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ]
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | retriever_filter,
        ),
        query_transform_prompt
        | openAIChatClient
        | StrOutputParser()
        | retriever_filter,
    ).with_config(run_name="chat_retriever_chain")

    document_chain = create_stuff_documents_chain(
        openAIChatClient, question_answering_prompt
    )
    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )

    chain_with_message_history = RunnableWithMessageHistory(
        conversational_retrieval_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="messages",
    )

    response = chain_with_message_history.invoke(
        {"input": user_question},
        {"configurable": {"session_id": session_id}},
    )

    pdf_dict = {}

    for doc in response["context"]:
        pdf_name = doc.metadata.get("source")
        page_num = doc.metadata.get("page_num")
        if pdf_name not in pdf_dict:
            pdf_url = next(
                (
                    pdf_info["pdf_url"]
                    for pdf_info in fetched_missing_pdfs
                    if pdf_info["pdf_name"] == pdf_name
                ),
                None,
            )
            pdf_dict[pdf_name] = {
                "pdf_name": pdf_name,
                "pdf_url": pdf_url,
                "pages": [],
            }
        pdf_dict[pdf_name]["pages"].append(page_num)

    pdfs_and_pages = list(pdf_dict.values())
    answer = response["answer"]
    if answer:
        chat_history.add_user_message(user_question)
        chat_history.add_ai_message(answer)

    return {"answer": answer, "pdfs_and_pages": pdfs_and_pages, "status": 200}


def conversation_chain(
    user_question,
    selected_pdf,
    qdrant_vector_embedding,
    cluster,
    user_name,
    session_id,
    token_expired,
    request_origin,
):
    try:

        retriever_filter = None
        fetched_missing_pdfs = fetch_missing_pdfs_from_firestore(
            cluster, session_id, token_expired
        )

        is_admin = (
            user_name == SUPER_ADMIN_USERNAME
            or cluster == UserClusters.ADMIN_CLUSTER.value
        )

        if selected_pdf != "":
            filter_kwargs = {"filter": {"source": selected_pdf}}
            if not is_admin:
                filter_kwargs["filter"]["group_id"] = cluster
        elif not is_admin:
            filter_kwargs = {"filter": {"group_id": cluster}}
        else:
            filter_kwargs = None

        if filter_kwargs:
            retriever_filter = qdrant_vector_embedding.as_retriever(
                search_kwargs={"k": 10, **filter_kwargs}
            )
        else:
            retriever_filter = qdrant_vector_embedding.as_retriever(
                search_kwargs={"k": 10}
            )

        # response = (
        #     langchain_conversation_with_history(
        #         user_question,
        #         session_id,
        #         chat_history,
        #         retriever_filter,
        #         fetched_missing_pdfs,
        #     )
        #     if request_origin == LOCAL_FRONT_END_URL
        #     else langchain_conversation_without_history(
        #         user_question, retriever_filter, fetched_missing_pdfs
        #     )
        # )

        response = open_ai_conversation_without_history(
            user_question, retriever_filter, fetched_missing_pdfs
        )

        return response

    except Exception as e:
        return {"status": 400, "error": str(e)}


def transcribe_audio(file_bytes, file_type, content_type):
    system_prompt = "You are a helpful assistant for an AI assisted chat bot that helps users search through clinical and medical guidelines. Accept the user question, correct any typographical errors and return the users exact words, Do not answer the questions, just return the exact question"

    file_buffer = io.BytesIO(file_bytes)
    file_info = ("temp." + file_type, file_buffer, content_type)
    transcript = openAIClient.audio.translations.create(
        model="whisper-1", file=file_info, response_format="text", prompt=MED_PROMPTS
    )

    corrected_transcript = openAIClient.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )

    return corrected_transcript.choices[0].message.content


def question_with_memory(user_question, session_id):
    try:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Your responsibility as an AI assistant is to analyze each new question in light of the ones previously posed, ensuring a clear understanding of the topic's development without altering the question's original intent. If the initial question is 'What is pemphigus vulgaris?' and the next inquiry evolves into 'What are the risk factors for developing this condition?', your task is to reflect this precise progression in your query formulation. Your objective is to construct a question that remains faithful to the user's latest question, maintaining the exact thematic focus and context. This approach demands that you preserve the essence and specific subject matter of the user's inquiry, resulting in a question like 'What are the risk factors for developing pemphigus vulgaris?'. It is crucial to ensure that your responses accurately mirror the user's questions, demonstrating a keen understanding of the topic's continuity or shifts without veering from the original question's scope.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        string_parser = StrOutputParser()

        runnable = question_prompt | openAIChatClient | string_parser

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in tavily_store:
                tavily_store[session_id] = ChatMessageHistory()
            return tavily_store[session_id]

        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        final_question = with_message_history.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}},
        )

        return final_question
    except Exception as e:
        return f"Error occurred: {str(e)}"


def extract_tavily_search_website_name_and_url(data):
    extracted_data = []

    for item in data:
        parsed_url = urlparse(item["url"])
        domain_name = parsed_url.netloc.replace("www.", "")
        extracted_data.append({"website_name": domain_name, "url": item["url"]})

    return extracted_data


def extract_serper_search_website_name_and_url(data):
    extracted_data = []

    for item in data.get("organic", []):
        parsed_url = urlparse(item["link"])
        domain_name = parsed_url.netloc.replace("www.", "")
        extracted_data.append({"website_name": domain_name, "url": item["link"]})

    return extracted_data


def serper_search(final_question):
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
    references = extract_serper_search_website_name_and_url(serper_data)

    prompt = [
        {
            "role": "system",
            "content": f"You are an AI-assisted medical doctor and research assistant with specialized expertise in medical sciences."
            f"Your primary function is to synthesize well-structured, critically analyzed, and medically accurate reports based on provided information."
            f"Your responses should emulate the communication style of a medical professional, incorporating appropriate medical terminology and considerations, and always adhere to the present simple tense for consistency",
        },
        {
            "role": "user",
            "content": f'Information: """{serper_response}"""\n\n'
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
    }  

    return response


def tavily_search(final_question):
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        tavily_response = client.search(
            query=final_question,
            search_depth="advanced",
            # include_domains=domains,
            max_results=10,
        )["results"]

        references = extract_tavily_search_website_name_and_url(tavily_response)
        prompt = [
            {
                "role": "system",
                "content": f"You are an AI-assisted medical doctor and research assistant with specialized expertise in medical sciences."
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
        answer = openAIChatClient.invoke(lc_messages).content
        response = {"answer": answer, "references": references}

        return response
    except Exception as e:
        return f"Error occurred: please try again later"
