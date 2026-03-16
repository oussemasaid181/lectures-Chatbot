from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
DATA_PATH = "data/info3"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
)

chunks = text_splitter.split_documents(load_documents())
document =chunks[10]
print(document.page_content)
print(document.metadata)

