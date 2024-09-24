"""Main script for running the RAG system with multi querying"""

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.utils.rag_components import (
    load_documents,
    split_documents,
    create_vectorstore,
    create_retriever,
    load_environment,
)


def create_multi_query_chain(llm):
    """Create a chain for generating multiple query variations."""
    TEMPLATE = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(TEMPLATE)
    return prompt_perspectives | llm | StrOutputParser() | (lambda x: x.split("\n"))


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def create_retrieval_chain(multi_query_chain, retriever):
    """Create a chain for document retrieval using multi-query approach."""
    return multi_query_chain | retriever.map() | get_unique_union


def create_rag_chain(retrieval_chain, llm):
    """Create the final RAG chain."""
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    return (
        {"context": retrieval_chain, "question": itemgetter("question")}
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
    question = "What is task decomposition for LLM agents?"

    # Indexing
    print("Loading documents...")
    docs = load_documents(url)

    print("Splitting documents...")
    splits = split_documents(docs)

    print("Creating vector store...")
    vectorstore = create_vectorstore(splits)

    print("Creating retriever...")
    retriever = create_retriever(vectorstore)

    # Create LLM
    llm = ChatOpenAI(temperature=0)

    # Create chains
    multi_query_chain = create_multi_query_chain(llm)
    retrieval_chain = create_retrieval_chain(multi_query_chain, retriever)
    rag_chain = create_rag_chain(retrieval_chain, llm)

    # Execute RAG chain
    answer = rag_chain.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
