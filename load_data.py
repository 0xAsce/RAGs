from langchain_community.document_loaders import DirectoryLoader

data_path = "data/books"

def load_data():
    loader = DirectoryLoader(data_path, glob="*.md")
    documents = loader.load()
    return documents
