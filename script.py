import argparse
import os
import shutil

from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "data/kommnetze"
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Shared embedding function to avoid duplication."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents() -> list[Document]:
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def save_to_chroma(chunks: list[Document]) -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=CHROMA_PATH,
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def ingest() -> None:
    """Load PDFs, split them, and store embeddings in Chroma."""
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)


def query(query_text: str) -> None:
    """Query the Chroma vector store and print matching context."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    results = db.similarity_search_with_score(query_text, k=3)

    if not results : #or results[0][1] < -1:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join(
        doc.page_content for doc, _score in results
    )
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    Answer:
    """

    pipe = pipeline(
       "text-generation",
         model="microsoft/phi-2",
        max_new_tokens=256,
    trust_remote_code=True,
    )
    model = HuggingFacePipeline(pipeline=pipe)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | model

    response = chain.invoke({"context": context_text, "question": query_text})
    print(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load PDFs and build the vector store")

    query_parser = subparsers.add_parser("query", help="Query the vector store")
    query_parser.add_argument("query_text", type=str, help="The query text")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "query":
        query(args.query_text)


if __name__ == "__main__":
    main()