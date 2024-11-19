import os
import uuid
from dotenv import load_dotenv
import psycopg
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as VectorStore
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from embedding_generator import EmbeddingGenerator  # Import your existing EmbeddingGenerator
from langchain_postgres import PostgresChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT", "5432") 


# basic connection string used with libraries like psycopg2
connection_url = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST} port={DB_PORT}"
# SQLAlchemy connection string used to define a connection for an ORM (Object-Relational Mapper) like SQLAlchemy.
connection_url_2 = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"



collection_name = "embeddings"
embedding_generator = EmbeddingGenerator()  # Initialize your embedding generator

def initialize_chat_history(session_id=None):
    """Initialize chat history with a new or existing session ID."""
    table_name = "chat_history"
    
    # Establish a synchronous connection to the database
    sync_connection = psycopg.connect(connection_url)
    
    # Create the chat history table if it doesn't exist
    PostgresChatMessageHistory.create_tables(sync_connection, table_name)
    
    # Generate a new session ID if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Initialize the PostgresChatMessageHistory instance
    chat_history = PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )
    
    return chat_history

def get_recent_chat_history(chat_history):
    """Retrieve the last 5 messages from chat history."""
    messages = chat_history.messages
    num_pairs = min(len(messages) // 2,5)
    return messages[-2 * num_pairs:]

def add_message_to_history(chat_history, role, content):
    """Add a message to the chat history based on the role."""
    if role == "system":
        message = SystemMessage(content=content)
    elif role == "ai":
        message = AIMessage(content=content)
    elif role == "human":
        message = HumanMessage(content=content)
    else:
        raise ValueError("Invalid role specified")
    
    # Add the message to the chat history
    chat_history.add_message(message)



def add_embedding_to_db(text_id, document):
    """Add an embedding to the vector store."""

    
    embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-small-en')
    vector_store = VectorStore(
        embeddings=embeddings,  
        collection_name=collection_name,
        connection=connection_url_2,
        use_jsonb=True,
    )
    # print(f"text:: {[text_content]}, {[text_id]}")
    vector_store.add_documents(documents=[document], ids=[text_id] )  # Store the embedding
    print(f"Added embedding for text_id: {text_id}")



def add_chat_to_db(chat_history, query, response):
    """Add a chat entry to the database."""
    # Add human query to chat history
    human_message = HumanMessage(content=query)
    chat_history.add_message(human_message)
    
    # Add AI response to chat history
    ai_message = AIMessage(content=response)
    chat_history.add_message(ai_message)
    
    print(f"Chat Entry Added - Query: {query}, Response: {response}")