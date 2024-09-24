# """Module providing a basic RAG example."""

# import bs4
# from dotenv import load_dotenv
# from langchain import hub
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# # Load environment variables from .env file
# # This will load all necessary API keys and configuration
# load_dotenv()

# #### INDEXING ####

# # Load Documents
# # WebBaseLoader is used to fetch and load the content from a specified URL
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# # Split documents into smaller chunks
# # This is crucial for effective retrieval, as it allows for more granular matching
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# # Create embeddings and store them in a vector database
# # Chroma is used as the vector store, and OpenAIEmbeddings for creating embeddings
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Create a retriever from the vector store
# # This will be used to fetch relevant documents based on the input query
# retriever = vectorstore.as_retriever()

# #### RETRIEVAL and GENERATION ####

# # Load a pre-defined prompt for RAG from LangChain's prompt hub
# prompt = hub.pull("rlm/rag-prompt")

# # Initialize the language model
# # We're using OpenAI's GPT-3.5-turbo model with temperature 0 for more deterministic outputs
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # Define a function to format retrieved documents
# # This function joins the content of all retrieved documents into a single string
# def format_docs(docs_to_format):
#     """joins the content of all retrieved documents into a single string"""
#     return "\n\n".join(doc.page_content for doc in docs_to_format)

# # Construct the RAG chain
# # This chain retrieves relevant documents, formats them, adds the query,
# # sends it all to the LLM, and then formats the output
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
#     | (lambda answer: f"\nAnswer:\n{answer}\n")
# )

# # Run an example query through the RAG chain
# QUESTION = "What is Task Decomposition?"
# answer = rag_chain.invoke(QUESTION)

# print(f"\nQuestion: {QUESTION}")
# print(answer)

# # Note: In a real application, you might want to add error handling,
# # allow for user input, and possibly cache results for efficiency.


"""Module providing a modular RAG (Retrieval-Augmented Generation) example."""

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

def main():
    """Main function to orchestrate the RAG process."""
    # Load environment variables
    load_environment()
    
    # Example URL and question
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    question = "What is Task Decomposition?"
    
    # Indexing
    print("Loading documents...")
    docs = load_documents(url)
    
    print("Splitting documents...")
    splits = split_documents(docs)
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(splits)
    
    print("Creating retriever...")
    retriever = create_retriever(vectorstore)
    
    # Retrieval and Generation
    print("Creating RAG chain...")
    rag_chain = create_rag_chain(retriever)
    
    # Run the query
    print(f"\nProcessing question: {question}")
    answer = rag_chain.invoke(question)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()