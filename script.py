from langchain.document_loaders import DirectoryLoader

DATA_PATH = "data/info3"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

