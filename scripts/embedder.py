import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def run_ingestion(source_file,persist_directory):
    """
    Transform a PDF into a searchable Vector Database.
    """
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Error: The source file '{source_file}' was not found. Embedding process is aborted.")
    

    print(f"---Starting Embedding Process for {source_file} ---")

    # clear all database if existed to update data when this script is run
    if os.path.exists(persist_directory):
        print(f"Resetting database: Removing existing directory at '{persist_directory}'...")
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            raise RuntimeError(f"Failed to clear old database directory: {str(e)}")

    try:
        loader = PyPDFLoader(source_file)
        documents = loader.load()

        # split in chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 100
        )

        chunks = text_splitter.split_documents(documents)

        # lightweight model (MiniLM) ==> runs fast on CPU + free
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

        # create DB
        Chroma.from_documents(
            documents = chunks,
            embedding = embeddings,
            persist_directory = persist_directory
        )

        print(f"Vector database created with success in {persist_directory}")

    except Exception as e: 
        raise RuntimeError(f"Error during embedding process: {str(e)}")

if __name__ == "__main__":
    source_file = ""
    persist_directory = ""
    run_ingestion(source_file,persist_directory)




