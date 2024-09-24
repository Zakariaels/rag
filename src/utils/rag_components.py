
"""Module providing components for the RAG (Retrieval-Augmented Generation) system."""
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def load_environment():
    """Load environment variables from .env file.
    
    This function uses python-dotenv to load environment variables,
    which typically include API keys and other configuration settings.
    """
    load_dotenv()

def load_documents(url):
    """Load documents from a given URL.
    
    Args:
        url (str): The URL to fetch content from.
    
    Returns:
        list: A list of loaded documents.
    
    This function uses WebBaseLoader to fetch and parse content from the specified URL.
    It focuses on specific HTML classes to extract relevant content.
    """
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks.
    
    Args:
        docs (list): List of documents to split.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The overlap between chunks in characters.
    
    Returns:
        list: A list of split documents.
    
    This function uses RecursiveCharacterTextSplitter to break documents into
    smaller, overlapping chunks for more effective retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def create_vectorstore(splits):
    """Create a vector store from document splits.
    
    Args:
        splits (list): List of document splits.
    
    Returns:
        Chroma: A Chroma vector store containing the document embeddings.
    
    This function creates embeddings for the document splits using OpenAIEmbeddings
    and stores them in a Chroma vector database for efficient retrieval.
    """
    return Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

def create_retriever(vectorstore):
    """Create a retriever from the vector store.
    
    Args:
        vectorstore (Chroma): The Chroma vector store.
    
    Returns:
        Retriever: A retriever object for fetching relevant documents.
    
    This function creates a retriever from the vector store, which will be used
    to fetch relevant documents based on input queries.
    """
    return vectorstore.as_retriever()

def format_docs(docs):
    """Format retrieved documents into a single string.
    
    Args:
        docs (list): List of retrieved documents.
    
    Returns:
        str: A string containing the content of all documents.
    
    This function joins the content of all retrieved documents into a single string,
    which can then be used as context for the language model.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, model_name="gpt-3.5-turbo", temperature=0):
    """Create the RAG (Retrieval-Augmented Generation) chain.
    
    Args:
        retriever (Retriever): The retriever object for fetching relevant documents.
        model_name (str): The name of the OpenAI model to use.
        temperature (float): The temperature setting for the language model.
    
    Returns:
        Chain: A LangChain chain that combines retrieval and generation.
    
    This function creates the complete RAG chain, which includes:
    1. Retrieving relevant documents
    2. Formatting the documents
    3. Combining the formatted documents with the input question
    4. Passing the combined input to the language model
    5. Parsing the output
    """
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )