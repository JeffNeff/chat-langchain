"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import GitLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os


def ingest_docs():
    """Get documents from web pages."""
    # ReadTheDocsLoader
    readthedocs_loader = ReadTheDocsLoader("docs.triggermesh.io/latest")
    raw_documents_rtd = readthedocs_loader.load()

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

    # Combine the documents from both sources
    raw_documents = raw_documents_rtd + raw_documents_git

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    ingest_docs()
