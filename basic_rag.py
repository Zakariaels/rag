import os
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
# This will load all necessary API keys and configuration
load_dotenv()

#### INDEXING ####

# Load Documents
# WebBaseLoader is used to fetch and load the content from a specified URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split documents into smaller chunks
# This is crucial for effective retrieval, as it allows for more granular matching
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create embeddings and store them in a vector database
# Chroma is used as the vector store, and OpenAIEmbeddings for creating embeddings
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

# Create a retriever from the vector store
# This will be used to fetch relevant documents based on the input query
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Load a pre-defined prompt for RAG from LangChain's prompt hub
prompt = hub.pull("rlm/rag-prompt")

# Initialize the language model
# We're using OpenAI's GPT-3.5-turbo model with temperature 0 for more deterministic outputs
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define a function to format retrieved documents
# This function joins the content of all retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construct the RAG chain
# This chain retrieves relevant documents, formats them, adds the query,
# sends it all to the LLM, and then formats the output
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | (lambda answer: f"\nAnswer:\n{answer}\n")
)

# Run an example query through the RAG chain
question = "What is Task Decomposition?"
answer = rag_chain.invoke(question)

print(f"\nQuestion: {question}")
print(answer)

# Note: In a real application, you might want to add error handling,
# allow for user input, and possibly cache results for efficiency.