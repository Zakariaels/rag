"""Main script for running the RAG system."""

from rag_components import (
    load_documents,
    split_documents,
    create_vectorstore,
    create_retriever,
    create_rag_chain,
    load_environment
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