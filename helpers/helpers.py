import os
import io
import uuid
import boto3
import hashlib
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from flask import jsonify
from qdrant.qdrant import (
    qdrant_vector_embedding,
)
from firestore.firestore import fetch_missing_pdfs_from_firestore
from helpers.constants import UserClusters
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.adapters.openai import convert_openai_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from tavily import TavilyClient
from langchain_core.messages import AIMessage, HumanMessage


load_dotenv()


tavily_store = {}
chat_history = []
openAIClient = OpenAI()
SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


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


def conversation_chain(
    user_question,
    selected_pdf,
    qdrant_vector_embedding,
    cluster,
    user_name,
    # chat_history,
):
    try:
        llm = openAIChatClient
        retriever_filter = None

        if (
            user_name == SUPER_ADMIN_USERNAME
            or cluster == UserClusters.ADMIN_CLUSTER.value
        ):
            if selected_pdf != "":
                retriever_filter = qdrant_vector_embedding.as_retriever(
                    search_kwargs={
                        "filter": {
                            "source": selected_pdf,
                        },
                        "k": 10,
                    }
                )
            else:
                retriever_filter = qdrant_vector_embedding.as_retriever()

        else:
            if selected_pdf != "":
                retriever_filter = qdrant_vector_embedding.as_retriever(
                    search_kwargs={
                        "filter": {
                            "source": selected_pdf,
                            "group_id": cluster,
                        },
                        "k": 10,
                    }
                )
            else:
                retriever_filter = qdrant_vector_embedding.as_retriever(
                    search_kwargs={
                        "filter": {
                            "group_id": cluster,
                        },
                        "k": 10,
                    }
                )

        user_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Generate a standalone question which is based on the question and the chat history. Create the standalone question without commentary. Always Analyze the given question and the chat history to identify a shift in conversation context, If a new question is introduced,which has no relationship with the chat history, generate a standalone question based on the new input. Ensure the standalone question is clear and independent, capable of standing alone without additional commentary",
                ),
            ]
        )

        system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful chatbot that responds in Markdown. I want you to act as an AI-assisted medical doctor that answers user's questions based on the context below providing very extensive responses.  Reemeber  to always list answers where appropriate for better readability and speak in present simple tense. It is very important to make sure that before providing a response, you always go through the following steps: 1. Think about what facts you need to look for in the context to answer the question? 2. Look for the details of the question in the context. 3. Does the context contain all the facts needed to answer the question? 4. Think about how you might answer the question given what you know. If you don’t have enough facts, answer with 'I don't know'  and do not try to come up with an answer. 5. Answer the question. \n\n {context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

        retriever_chain = create_history_aware_retriever(
            llm, retriever_filter, user_prompt
        )
        stuff_documents_chain = create_stuff_documents_chain(llm, system_prompt)
        chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
        response = chain.invoke(
            {
                "chat_history": chat_history,
                "input": user_question,
            }
        )
        page_nums = [doc.metadata.get("page_num") for doc in response["context"]]
        pdf_names = [doc.metadata.get("source") for doc in response["context"]]

        pdfs_and_pages = []
        pdf_dict = {}

        def find_pdf_url(pdf_data, target_pdf_name):
            for pdf_info in pdf_data:
                if pdf_info["pdf_name"] == target_pdf_name:
                    return pdf_info["pdf_url"]
            return None

        cluster = "admin_cluster"
        user_name = "yahayakenny"
        fetched_missing_pdfs = fetch_missing_pdfs_from_firestore(cluster, user_name)

        for pdf_name, page_num in zip(pdf_names, page_nums):
            if pdf_name not in pdf_dict:
                pdf_url = find_pdf_url(fetched_missing_pdfs, pdf_name)
                pdf_dict[pdf_name] = {
                    "pdf_name": pdf_name,
                    "pdf_url": pdf_url,
                    "pages": [],
                }
            pdf_dict[pdf_name]["pages"].append(page_num)

        pdfs_and_pages = list(pdf_dict.values())
        answer = response["answer"]

        if answer:
            chat_history.append(HumanMessage(content=user_question))
            chat_history.append(AIMessage(content=answer))

            # return jsonify(chain_response), chain_response.get("status", 200)

        return {"answer": answer, "pdfs_and_pages": pdfs_and_pages, "status": 200}

    except Exception as e:
        return {
            "status": 400,
        }


def transcribe_audio(file_bytes, file_type, content_type):
    system_prompt = "You are a helpful assistant for an AI assisted chat bot that helps users search through clinical and medical guidelines. Accept the user question, correct any typographical errors and return the users exact words, Do not answer the questions, just return the exact question"

    file_buffer = io.BytesIO(file_bytes)
    file_info = ("temp." + file_type, file_buffer, content_type)
    transcript = openAIClient.audio.transcriptions.create(
        model="whisper-1",
        file=file_info,
        response_format="text",
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


def question_with_memory(user_question):
    try:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Imagine you're an AI assistant tasked with understanding the relationship between two questions. Given an example initial question like 'What is gout?' and the follow-up question 'How is it managed?', your role is to generate a new question that encapsulates the shared context between both questions and provide a final response like 'how is gout managed'. Your response should reflect an understanding of the topic introduced in the initial question and the specific aspect addressed in the follow-up question. Keep the generated question concise and relevant. If the new question can stand alone, then return this as the response",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        string_parser = StrOutputParser()
        # json_parser = JsonOutputParser()

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
            config={"configurable": {"session_id": "abc123"}},
        )

        return final_question
    except Exception as e:
        return f"Error occurred: {str(e)}"


def tavily_search(final_question):
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        tavily_response = client.search(query=final_question, search_depth="advanced")[
            "results"
        ]

        prompt = [
            {
                "role": "system",
                "content": f"You are an AI critical thinker research assistant. "
                f"Your sole purpose is to write well written, critically acclaimed,"
                f"objective and structured reports on given text.",
            },
            {
                "role": "user",
                "content": f'Information: """{tavily_response}"""\n\n'
                f"Using the above information, answer the following"
                f'query: "{final_question}" in a detailed report --'
                f"Please use MLA format and markdown syntax.",
            },
        ]

        lc_messages = convert_openai_messages(prompt)
        report = openAIChatClient.invoke(lc_messages).content

        return report
    except Exception as e:
        return f"Error occurred: {str(e)}"
