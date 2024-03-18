import os
import io
import uuid
import boto3
import hashlib
from urllib.parse import urlparse
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
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient
from langchain_core.messages import AIMessage, HumanMessage


load_dotenv()
tavily_store = {}
chat_history = {}
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

domains = [
    "https://www.nice.org.uk/guidance",
    "https://cks.nice.org.uk/",
    "https://dermnetnz.org/",
    "https://www.pcds.org.uk/",
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
    "https://pubmed.ncbi.nlm.nih.gov/",
    "https://www.cochranelibrary.com/",
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


def conversation_chain(
    user_question, selected_pdf, qdrant_vector_embedding, cluster, user_name, session_id, token_expiry
):
    try:

        llm = openAIChatClient
        retriever_filter = None
     
        fetched_missing_pdfs = fetch_missing_pdfs_from_firestore(cluster, session_id,token_expiry)

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

        user_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Your responsibility as an AI assistant is to analyze each new question in light of the ones previously posed, ensuring a clear understanding of the topic's development without altering the question's original intent. If the initial question is 'What is pemphigus vulgaris?' and the next inquiry evolves into 'What are the risk factors for developing this condition?', your task is to reflect this precise progression in your query formulation. Your objective is to construct a question that remains faithful to the user's latest question, maintaining the exact thematic focus and context. This approach demands that you preserve the essence and specific subject matter of the user's inquiry, resulting in a question like 'What are the risk factors for developing pemphigus vulgaris?'. It is crucial to ensure that your responses accurately mirror the user's questions, demonstrating a keen understanding of the topic's continuity or shifts without veering from the original question's scope.",
                ),
            ]
        )

        system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "As an AI-assisted medical doctor, your objective is to provide responses that mirror the precision and professionalism of medical communication. Utilize use MLA format and Markdown for clarity and organization, ensuring your answers to user inquiries are thorough and reflect medical expertise. Adhere to the present simple tense for consistency. Before responding, meticulously follow these steps to ensure accuracy and relevance: 1.Identify Necessary Information: Consider what medical facts or details are crucial for addressing the user's question based on the provided context. 2. Examine the Question Details: Scrutinize the context for specifics related to the user's inquiry.3. Assess Fact Availability: Determine if the context includes all necessary information to formulate a comprehensive answer. 4.Formulate Your Response: Based on the available facts, contemplate the most accurate and informative answer. If the context lacks sufficient information, respond with 'I don't know' rather than speculating.5.Provide the Answer: Answer the question with detailed explanations, listing answers where appropriate for enhanced readability.Remember, your responses should not only convey medical knowledge but also uphold the professionalism expected in medical dialogues.{context}",
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
                "chat_history": chat_history.get(session_id, []),
                "input": user_question,
            }
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
            if session_id not in chat_history:
                chat_history[session_id] = []
            chat_history[session_id].append(HumanMessage(content=user_question))
            chat_history[session_id].append(AIMessage(content=answer))

        return {"answer": answer, "pdfs_and_pages": pdfs_and_pages, "status": 200}

    except Exception as e:
        return {"status": 400, "error": str(e)}


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


def extract_website_name_and_url(data):
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
            max_results=5,
        )["results"]
        references = extract_website_name_and_url(tavily_response)
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
