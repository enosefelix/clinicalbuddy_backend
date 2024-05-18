import json
import os
import time
from datetime import datetime
from firestore.firestore import fetch_missing_pdfs_from_firestore
from config.constants import UserClusters
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from helpers.conversation_helpers import (
    grade_docs_with_openai,
    reciprocal_rank_fusion,
    grade_docs_with_cohere,
    generate_final_response_with_openai,
    serper_search,
    openAIChatClient,
)

SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME")


def conversation(
    user_question, question_history, retriever_filter, fetched_missing_pdfs
):
    filtered_relevant_ranked_data = []
    filtered_not_relevant_ranked_data = []

    # Measure the start time
    start_time_total = time.time()
    print(
        "Start time for the entire process:",
        datetime.fromtimestamp(start_time_total).strftime("%I:%M %p"),
    )

    # Measure the start time for retriever chain
    start_time_for_retriever = time.time()
    print(
        "Start time for retriever:",
        datetime.fromtimestamp(start_time_for_retriever).strftime("%I:%M %p"),
    )

    # Using a combination of Multiquery retriever + HyDE document generation
    template = """You are a helpful assistant, your task is to generate 2  variations of the user's question looking at the question from different perspectives. Then provide concise but accurate answers to  the generated question which demonstrates medical expertise,  \n OUTPUT (2 answers):
    Question: {question}"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    # print("prompt hyde >>", prompt_hyde)

    generated_answers = (
        prompt_hyde | openAIChatClient | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # print("generated answers >>", generated_answers)

    # Using rag fusion to rerank order of retrieved documents
    reranked_retriever_chain = (
        generated_answers | retriever_filter.map() | reciprocal_rank_fusion
    )
    reranked_data = reranked_retriever_chain.invoke({"question": user_question})

    if reranked_data:
        end_time_for_retriever = time.time()
        print(
            "End time for retriever:",
            datetime.fromtimestamp(end_time_for_retriever).strftime("%I:%M %p"),
        )
        elapsed_time_for_retriever = end_time_for_retriever - start_time_for_retriever
        print(
            "Elapsed time for retriever: {:.2f} seconds".format(
                elapsed_time_for_retriever
            )
        )

    # Measure the start time for checking document relavance
    start_time_document_relevance = time.time()
    print(
        "Start time for checking document relavance:",
        datetime.fromtimestamp(start_time_document_relevance).strftime("%I:%M %p"),
    )

    def check_document_relevance(data):
        prompt_rag = f"""You are a grader assessing relevance of a retrieved document to a user question. \n
            Here is the retrieved document: \n\n {data} \n\n
            Here is the user question: {user_question} \n
            If the document contains keywords or phrases directly related to the user question, grade it as relevant. \n
            It needs to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. Only ouput the object without any text.
        """

        try:
            return grade_docs_with_openai(prompt_rag=prompt_rag)

        except Exception as e:
            raise Exception(f"An error occurred while checking relevance: {e}")

    def grade_documents(data_list):
        for i, document_data in enumerate(data_list, start=1):
            page_content = document_data[0].page_content
            try:
                res = check_document_relevance(page_content)
                if res is not None:
                    res_json = json.loads(res)  # Parse the response as JSON
                    if res_json.get("score") == "yes":
                        print(f"---GRADE: DOCUMENT {i} RELEVANT---")
                        filtered_relevant_ranked_data.append(document_data)
                    else:
                        print(f"---GRADE: DOCUMENT {i} NOT RELEVANT---")
                        filtered_not_relevant_ranked_data.append(document_data)

            except Exception as e:
                raise Exception(f"An error occurred while grading documents: {e}")

    if reranked_data:
        grade_documents(reranked_data)

    if filtered_relevant_ranked_data and filtered_not_relevant_ranked_data:
        # Measure the end time for OpenAI calls
        end_time_document_relevance = time.time()
        print(
            "End time for checking document relevance:",
            datetime.fromtimestamp(end_time_document_relevance).strftime("%I:%M %p"),
        )
        elapsed_time_document_relevaance = (
            end_time_document_relevance - start_time_document_relevance
        )
        print(
            "Elapsed time for document relevance: {:.2f} seconds".format(
                elapsed_time_document_relevaance
            )
        )

    pdf_dict = {}

    if filtered_relevant_ranked_data:
        for doc, _ in filtered_relevant_ranked_data:
            pdf_name = doc.metadata.get("source")
            page_num = doc.metadata.get("page_num")
            if pdf_name not in pdf_dict:
                pdf_info = next(
                    (
                        pdf_info
                        for pdf_info in fetched_missing_pdfs
                        if pdf_info["pdf_name"] == pdf_name
                    ),
                    None,
                )
                if pdf_info:
                    pdf_url = pdf_info["pdf_url"]
                else:
                    pdf_url = None
                pdf_dict[pdf_name] = {
                    "pdf_name": pdf_name,
                    "pdf_url": pdf_url,
                    "pages": [],
                }
            pdf_dict[pdf_name]["pages"].append(page_num)

    pdfs_and_pages = list(pdf_dict.values())

    start_time_final_response = time.time()
    print(
        "Start time for providing a final response:",
        datetime.fromtimestamp(start_time_final_response).strftime("%I:%M %p"),
    )

    if len(filtered_relevant_ranked_data) > 0:
        print(
            "Filtered relevant ranked data found. Proceeding with generating response."
        )
        try:

            final_response = generate_final_response_with_openai(
                filtered_relevant_ranked_data, user_question
            )

            end_time_total = time.time()
            print(
                "End time for the entire process:",
                datetime.fromtimestamp(end_time_total).strftime("%I:%M %p"),
            )

            # Calculate the total elapsed time for the entire process
            elapsed_time_total = end_time_total - start_time_total
            print("Total elapsed time: {:.2f} seconds".format(elapsed_time_total))
            return {
                "answer": final_response,
                "pdfs_and_pages": pdfs_and_pages[:6],
                "status": 200,
                "source": "knowledge_base",
            }
        except Exception as e:
            print(f"An error occurred while generating response: {e}")
            return {
                "answer": "An error occurred while generating response.",
                "pdfs_and_pages": [],
                "status": 500,
            }
    else:
        response = serper_search(user_question)

        if response:
            return response


def conversation_chain(
    user_question,
    question_history,
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
                search_kwargs={"k": 5, **filter_kwargs}
            )
        else:
            retriever_filter = qdrant_vector_embedding.as_retriever(
                search_kwargs={"k": 5}
            )

        response = conversation(
            user_question, question_history, retriever_filter, fetched_missing_pdfs
        )

        return response

    except Exception as e:
        return {"status": 400, "error": str(e)}
