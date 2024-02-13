import os
import qdrant_client
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import  Qdrant
from langchain_openai import OpenAIEmbeddings


from qdrant_client.http import models
from dotenv import load_dotenv


load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Create Qdrant client
qd_client = qdrant_client.QdrantClient(QDRANT_HOST, api_key=QDRANT_API_KEY)

# create collection
collection_config = qdrant_client.http.models.VectorParams(
    size=1536, distance=qdrant_client.http.models.Distance.COSINE
)


# Create Qdrant vector embedding
embedding = OpenAIEmbeddings()
qdrant_vector_embedding = Qdrant(
    client=qd_client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embedding,
)


def create_or_get_collection(
    qd_client,
    QDRANT_COLLECTION_NAME,
    collection_config,
):
    try:
        # Specify vector parameters when creating the collection
        collection = qd_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=collection_config,
        )

        return collection
    except Exception as e:
        return None


def delete_selected_pdf_from_qdrant(pdf_name, user_name):
    qd_client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=pdf_name),
                    ),
                    models.FieldCondition(
                        key="metadata.user_name",
                        match=models.MatchValue(value=user_name),
                    ),
                ],
            ),
        ),
    )


def delete_all_selected_pdf_array_from_qdrant(selected_pdfs, user_name):
    qd_client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchAny(any=selected_pdfs),
                    ),
                    models.FieldCondition(
                        key="metadata.user_name",
                        match=models.MatchValue(value=user_name),
                    ),
                ],
            ),
        ),
    )


def delete_all_pdfs_by_user_from_qdrant(user_name):
    qd_client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.user_name",
                        match=models.MatchValue(value=user_name),
                    ),
                ],
            ),
        ),
    )
