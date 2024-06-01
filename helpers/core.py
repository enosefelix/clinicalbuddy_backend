import os
import json
from firestore.firestore import fetch_missing_pdfs_from_firestore
from config.constants import UserClusters, LOCAL_FRONT_END_URL
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_transformers import (
    LongContextReorder,
)
from helpers.conversation_helpers import (
    grade_docs_with_openai,
    reciprocal_rank_fusion,
    grade_docs_with_cohere,
    generate_final_response_with_cohere,
    serper_search,
    openAIChatClient,
    generate_response_with_instructor_openai,
)


SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME")

def conversation(
    user_question,
    question_history,
    retriever_filter,
    fetched_missing_pdfs,
    request_origin,
    selected_pdf,
):

    filtered_relevant_ranked_data_all = []
    filtered_relevant_ranked_data_selected = []

    def check_document_relevance(page_content, user_question, request_origin):
        prompt_rag = f"""You are a grader assessing relevance of a retrieved document to a user question. \n
            Here is the retrieved document: \n\n {page_content} \n\n
            Here is the user question: {user_question} \n
            If the document contains keywords or phrases directly related to the user question, grade it as relevant. \n
            It doesnt need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. Only ouput the object without any text.
        """

        try:
            if request_origin == LOCAL_FRONT_END_URL:
                return grade_docs_with_cohere(prompt_rag=prompt_rag)
            else:
                return grade_docs_with_openai(prompt_rag=prompt_rag)

        except Exception as e_openai:
            try:
                return grade_docs_with_openai(prompt_rag=prompt_rag)
            except Exception as e_cohere:
                return f"Error while grading documents { e_cohere, e_openai}"

    def grade_documents(reranked_data, user_question, request_origin, selected):
        for i, document_data in enumerate(reranked_data, start=1):
            page_content = None
            if selected == True:
                page_content = document_data.page_content
            else:
                page_content = document_data[0].page_content
            try:
                res = check_document_relevance(
                    page_content, user_question, request_origin
                )
                if res is not None:
                    res_json = json.loads(res)
                    if res_json.get("score") == "yes":
                        print(f"---GRADE: DOCUMENT {i} RELEVANT---")
                        if selected == True:
                            filtered_relevant_ranked_data_selected.append(document_data)
                        else:
                            filtered_relevant_ranked_data_all.append(document_data)
                    else:
                        print(f"---GRADE: DOCUMENT {i} NOT RELEVANT---")

            except Exception as e:
                raise Exception(f"An error occurred while grading documents: {e}")

    def generate_response_with_citations(
        user_question, filtered_relevant_ranked_data, fetched_missing_pdfs
    ):
        # Generate the response using the provided method
        response_dict = generate_response_with_instructor_openai(
            user_question, filtered_relevant_ranked_data
        )
        answer = response_dict.get("answer")
        citations = response_dict.get("citations")
        citation_dict = {}

        # Process the citations

        if citations:
            for citation in citations:
                source = citation.get("source")
                page_num = citation.get("page_num")
                quote = citation.get("quote")

                if source:
                    pdf_name = source
                    pdf_info = next(
                        (
                            pdf_info
                            for pdf_info in fetched_missing_pdfs
                            if pdf_info["pdf_name"] == pdf_name
                        ),
                        None,
                    )
                    pdf_url = pdf_info["pdf_url"] if pdf_info else None

                    if pdf_name not in citation_dict:
                        citation_dict[pdf_name] = {
                            "pdf_name": pdf_name,
                            "quote_and_pages": [],
                        }

                    if not any(
                        entry["quote"] == quote and entry["page"] == page_num
                        for entry in citation_dict[pdf_name]["quote_and_pages"]
                    ):
                        citation_dict[pdf_name]["quote_and_pages"].append(
                            {
                                "quote": quote,
                                "page": page_num,
                                "pdf_url": pdf_url,
                            }
                        )

        citation_and_pages = [
            {"pdf_name": key, "quote_and_pages": value["quote_and_pages"]}
            for key, value in citation_dict.items()
        ]

        return answer, citation_and_pages

    def extract_pdfs_and_pages(filtered_relevant_ranked_data, fetched_missing_pdfs):
        pdf_dict = {}

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
                pdf_url = pdf_info["pdf_url"] if pdf_info else None

                pdf_dict[pdf_name] = {
                    "pdf_name": pdf_name,
                    "pdf_url": pdf_url,
                    "pages": [],
                }

            pdf_dict[pdf_name]["pages"].append(page_num)

        pdfs_and_pages = list(pdf_dict.values())
        return pdfs_and_pages

    if selected_pdf == "":
        citation_and_pages = []
        pdfs_and_pages = []

        # Using a combination of Multiquery retriever + HyDE document generation
        template = """You are a helpful research assistant, your task is to generate 2  variations of the user's question looking at the question from different perspectives. Then write a scientific paper that provides concise but accurate answers to the generated question which demonstrates medical expertise,  \n OUTPUT (2 answers):
        Question: {question}"""
        prompt_hyde = ChatPromptTemplate.from_template(template)

        generated_answers = (
            prompt_hyde
            | openAIChatClient
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        print("gen answers", generated_answers)

        # Using rag fusion to rerank order of retrieved documents
        retriever_chain = (
            generated_answers | retriever_filter.map() | reciprocal_rank_fusion
        )
        retrieved_data = retriever_chain.invoke({"question": user_question})
        reordering = LongContextReorder()
        reranked_data = reordering.transform_documents(retrieved_data)

        if reranked_data:
            grade_documents(reranked_data, user_question, request_origin, False)
            print("finished grading documents>>>")

            if filtered_relevant_ranked_data_all:
                try:
                    pdfs_and_pages = extract_pdfs_and_pages(
                        filtered_relevant_ranked_data_all, fetched_missing_pdfs
                    )

                    answer, citation_and_pages = generate_response_with_citations(
                        user_question,
                        filtered_relevant_ranked_data_all,
                        fetched_missing_pdfs,
                    )

                except Exception as e_openai:
                    try:
                        answer = generate_final_response_with_cohere(
                            filtered_relevant_ranked_data_all, user_question
                        )
                    except Exception as e_cohere:
                        return {
                            "answer": "An error occurred while generating response with both OpenAI and Cohere.",
                            "pdfs_and_pages": [],
                            "status": 500,
                        }

                return {
                    "answer": answer,
                    "pdfs_and_pages": pdfs_and_pages[:6],
                    "citation_and_pages": citation_and_pages,
                    "status": 200,
                    "source": "knowledge_base",
                }

            else:
                print(">>> Commencing web search")
                response = serper_search(user_question)
                if response:
                    print(">>> finisehed web search")
                    return response
                else:
                    return {
                        "answer": "No relevant data found and search did not yield results.",
                        "pdfs_and_pages": [],
                        "status": 404,
                    }

    else:
        citation_and_pages = []
        retrieved_data = retriever_filter.invoke(user_question)
        reordering = LongContextReorder()
        reranked_data = reordering.transform_documents(retrieved_data)

        if reranked_data:
            grade_documents(reranked_data, user_question, request_origin, True)

        if filtered_relevant_ranked_data_selected:
            try:
                answer, citation_and_pages = generate_response_with_citations(
                    user_question,
                    filtered_relevant_ranked_data_selected,
                    fetched_missing_pdfs,
                )

            except Exception as e_openai:
                try:
                    answer = generate_final_response_with_cohere(
                        filtered_relevant_ranked_data_selected, user_question
                    )
                except Exception as e_cohere:
                    return {
                        "answer": "An error occurred while generating response with both OpenAI and Cohere.",
                        "pdfs_and_pages": [],
                        "status": 500,
                    }

            return {
                "answer": answer,
                "pdfs_and_pages": [],
                "citation_and_pages": citation_and_pages,
                "status": 200,
                "source": "knowledge_base",
            }

        else:
            print(">>> Commencing web search")
            response = serper_search(user_question)
            if response:
                print(">>> finisehed web search")
                return response
            else:
                return {
                    "answer": "No relevant data found and search did not yield results.",
                    "pdfs_and_pages": [],
                    "status": 404,
                }

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
            cluster,
            session_id,
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
                search_kwargs={"k": 5, "score_threshold": 0.8, **filter_kwargs},
                search_type="similarity_score_threshold",
            )

        else:
            retriever_filter = qdrant_vector_embedding.as_retriever(
                search_kwargs={"k": 5, "score_threshold": 0.8},
                search_type="similarity_score_threshold",
            )

        response = conversation(
            user_question,
            question_history,
            retriever_filter,
            fetched_missing_pdfs,
            request_origin,
            selected_pdf,
        )

        return response

    except Exception as e:
        return {"status": 400, "error": str(e)}