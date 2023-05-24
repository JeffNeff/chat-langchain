import os
import glob
import pickle

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import GitLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os

def ingest_docs():
    """Get documents from web pages."""
    html_files = glob.glob("/home/jeff/chat-langchain/docs.triggermesh.io/1.25/**/*.html", recursive=True)
    if not html_files:
        print("No HTML files found in the specified directory.")
        return

    documents = []
    for file_path in html_files:
        loader = UnstructuredHTMLLoader(file_path)
        raw_documents = loader.load()
        documents.extend(raw_documents)

    # GitLoader
    repo_path = "repo_path"
    clone_url = "https://github.com/triggermesh/triggermesh.git"

    if os.path.exists(repo_path):
        git_loader = GitLoader(repo_path)
        local_path = repo_path
    else:
        git_loader = GitLoader(repo_path, clone_url=clone_url)
        local_path = git_loader.repo_path

    raw_documents_git = git_loader.load()

    raw_documents = documents + raw_documents_git

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    split_documents = text_splitter.split_documents(raw_documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
