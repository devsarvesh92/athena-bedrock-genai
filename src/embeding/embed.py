from langchain_community.embeddings import BedrockEmbeddings

from langchain_core.documents import Document
import boto3
from langchain.vectorstores.pgvector import PGVector

bedrock_client = boto3.client("bedrock-runtime")

bedrock_embedings: BedrockEmbeddings = BedrockEmbeddings(
    client=bedrock_client, model_id="amazon.titan-embed-text-v1"
)

CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/vector_db"
)


def generate_and_store_embedings(
    *, documents: list[Document], collection_name: str
) -> None:
    PGVector.from_documents(
        embedding=bedrock_embedings,
        documents=documents,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
    )


def get_similar_documents(*, prompt: str, collection_name: str) -> list[Document]:
    vectorstore = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=bedrock_embedings,
        collection_name=collection_name,
        use_jsonb=True,
    )
    return vectorstore.similarity_search(query=prompt)


if __name__ == "__main__":
    from langchain_community.document_loaders import JSONLoader

    # Load schema
    generate_and_store_embedings(
        documents=JSONLoader(
            file_path="rag_dataset/schema.json",
            jq_schema=".schemas[].content",
            text_content=False,
        ).load(),
        collection_name="table_schema",
    )

    # Load examples
    generate_and_store_embedings(
        documents=JSONLoader(
            file_path="rag_dataset/valid_examples.json",
            jq_schema=".examples[].content",
            text_content=False,
        ).load(),
        collection_name="query_valid_examples",
    )

    generate_and_store_embedings(
        documents=JSONLoader(
            file_path="rag_dataset/error_examples.json",
            jq_schema=".examples[].content",
            text_content=False,
        ).load(),
        collection_name="query_invalid_examples",
    )
