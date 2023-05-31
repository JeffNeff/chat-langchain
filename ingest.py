"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import GitLoader
# from langchain.document_loaders import YoutubeLoader
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

    # Additional repositories
    repositories = [
        {"repo_path": "repo_path/core", "clone_url": "https://github.com/triggermesh/triggermesh-core"},
        {"repo_path": "repo_path/scoby", "clone_url": "https://github.com/triggermesh/scoby"},
        {"repo_path": "repo_path/tmctl", "clone_url": "https://github.com/triggermesh/tmctl"},
        {"repo_path": "repo_path/brokers", "clone_url": "https://github.com/triggermesh/brokers"},
        {"repo_path": "repo_path/aws-custom-runtime", "clone_url": "https://github.com/triggermesh/aws-custom-runtime"},
    ]

    raw_documents_git = []

    for repo in repositories:
        if os.path.exists(repo['repo_path']):
            git_loader = GitLoader(repo['repo_path'])
            local_path = repo['repo_path']
        else:
            git_loader = GitLoader(repo['repo_path'], clone_url=repo['clone_url'])
            local_path = git_loader.repo_path

        raw_documents_git.extend(git_loader.load())

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
