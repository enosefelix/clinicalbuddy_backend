import re
from pydantic import BaseModel, Field, validator
from typing import List, Dict


class Citation(BaseModel):
    source: str = Field(
        description="The name of the pdf document from which the answer was obtained. Make sure this is only from the sources provided. Please do not come up with a source from your innate knowledge",
    )
    quote: str = Field(
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )
    page_num: int = Field(
        description="The page number of the pdf document from which the answer was obtained.",
    )


class Fact(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used. If none of the documents answer the question, just say you don't know.Never add reference or citations to the answer."""

    answer: str = Field(
        description="This is the answer to the question. Answer the user question based ONLY on the given sources, Always utilize use MLA format and Markdown for clarity and organization, ensuring your answers are thorough and reflect medical expertise. Adhere to the present simple tense for consistency. Answer the question with detailed explanations, listing and highlighting answers where appropriate for enhanced readability. Never add reference or citations to the answer. ",
    )
    citations: List[Citation] = Field(default_factory=list)

    @validator("citations", pre=True, always=True)
    def validate_sources(cls, v, values, **kwargs):
        context = kwargs.get("context", None)
        if context:
            documents = context.get(
                "text_chunk", None
            )  # List of tuples (Document, similarity_score)
            if documents:
                new_citations = []
                for doc_tuple in documents:
                    document, _ = doc_tuple  # Unpack the tuple
                    page_content = document.page_content
                    metadata = document.metadata
                    source = metadata.get("source")
                    page_num = metadata.get("page_num")
                    spans = list(cls.get_spans(cls, page_content, values["answer"]))
                    for span in spans:
                        quote = page_content[span[0] : span[1]]
                        new_citations.append(
                            Citation(source=source, quote=quote, page_num=page_num)
                        )

                return new_citations
        return v

    def get_spans(self, context: str, answer: str):
        for quote in [answer]:
            yield from self._get_span(quote, context)

    def _get_span(self, quote: str, context: str):
        for match in re.finditer(re.escape(quote), context):
            yield match.span()


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, str]):
        self.page_content = page_content
        self.metadata = metadata
