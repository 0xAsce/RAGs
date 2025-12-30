from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

data_path = "data/books"

def load_data():
    loader = DirectoryLoader(data_path, glob="*.md")
    documents = loader.load()
    return documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function = len,
    add_start_index = True,
)
chunk = text_splitter.split_documents(load_data())

document = chunk[10]
print(document.page_content)
print(document.metadata)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

chroma_path = "chroma"
db = Chroma.from_documents(
    chunk, embeddings, persist_directory=chroma_path
)