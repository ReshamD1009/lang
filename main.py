import os
from flask import Flask, request, render_template, session
from database import initialize_chat_history, add_chat_to_db, get_recent_chat_history, add_embedding_to_db  # Import functions from database.py
from embedding_generator import EmbeddingGenerator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import uuid
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__,template_folder='.')
app.secret_key = os.urandom(24)  

DATA_FILE_PATHS = [
    'C:/Users/Coditas-Admin/Desktop/vinod_langchain/example.pdf'
]

def load_text_samples(file_paths):
    
    # empty list documents to store the processed Document objects.
    documents =[]
    for file_path in file_paths: 
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{file_path}' not found.")
        
        loader = PyPDFLoader(file_path)
        # loader.load() to extract the text from the PDF
        pdf_documents = loader.load()
        

        for doc in pdf_documents:
            # Wraps the page_content of each chunk in a Document object.
            document = Document(page_content=doc.page_content)
            documents.append(document)

        return documents
    

def process_embeddings(documents):
    
    if not documents:
        print("No documents to process")
        return
    for document in documents:
        text_id = str(uuid.uuid4()) #universally unique identifier (each document has diff text_id)
        try:
            add_embedding_to_db(text_id,document)
        except Exception as e:
            print(f"Error processing document:{e}")
       

def generate_augmented_response(query: str, retrieved_items: List[tuple[str,str]], last_five_context: str): #o/p-dictionary of query and generated response
    text_splitter = CharacterTextSplitter(
        chunk_size=400,  
        chunk_overlap=100
    )

    all_chunks = []
    for idx, (text, second_text) in enumerate(retrieved_items):  
        chunks = text_splitter.split_text(text)  
        all_chunks.extend(chunks)

# create unique text_id for each chunk
# idx - index of document & chunks position in chunks
        for chunk in chunks:
            text_id = f"{idx}_{chunks.index(chunk)}"
            document = Document(page_content=chunk)
            add_embedding_to_db(text_id, document)  

        second_chunks = text_splitter.split_text(second_text)  
        for second_chunk in second_chunks:
            second_text_id = f"{idx}_second_{second_chunks.index(second_chunk)}"
            document = Document(page_content=second_chunk)
            add_embedding_to_db(second_text_id, document)  

    context = f"{last_five_context}\n\n" + "\n\n".join(f"Document {idx + 1}:\n{chunk}" for idx, chunk in enumerate(all_chunks))
    
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="mixtral-8x7b-32768",  
        temperature=0.5,
        max_tokens=1000,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that provides comprehensive answers by analyzing and synthesizing information from multiple documents.",
            ),
            ("human", f"Using the following context, please provide a comprehensive response to the question:\n\nContext:\n{context}\n\nQuestion: {query}"),
        ]
    )

    chain = prompt | llm
    ai_msg = chain.invoke({
        "input_language": "English",  
        "output_language": "English",  
        "input": query, 
    }) 

    response = ai_msg.content.strip()  
    return {
        "query": query,
        "generated_response": response
    }

@app.route('/', methods=['GET', 'POST'])
def index():

# session.get: Retrieves the current session ID.
# initialize_chat_history: Either loads an existing chat history using the session ID or initializes a new one.
# Saves the session_id back to the session object to maintain continuity.

    session_id = session.get('session_id', None)
    chat_history = initialize_chat_history(session_id)  
    session['session_id'] = chat_history._session_id  # Save session ID (accessing private attribute)

    try:
        if request.method == 'POST':
            query_text = request.form['query']

            texts = load_text_samples(DATA_FILE_PATHS)
            process_embeddings(texts)  

            last_five_chats = get_recent_chat_history(chat_history)  
            
            last_five_context = "\n\n".join(
                f"User: {chat.content}\nResponse: {chat.additional_kwargs.get('response', 'No response')}"
                for chat in last_five_chats
            )

            retrieved_items = [(msg.content, "") for msg in last_five_chats]  # Use chat content for retrieval

            result = generate_augmented_response(query_text, retrieved_items, last_five_context)

            
            add_chat_to_db(chat_history, query_text, result['generated_response'])  

            last_five_chats = get_recent_chat_history(chat_history)  # Update chat history in-memory

            formatted_chat_history = []
            for i in range(0, len(last_five_chats), 2):
                human_message = last_five_chats[i]
                ai_message = last_five_chats[i + 1] if i + 1 < len(last_five_chats) else AIMessage(content="No response yet")
                formatted_chat_history.append({
                    'query': human_message.content,
                    'response': ai_message.content
                })
            return render_template('chatbot.html',query =query_text, result = result['generated_response'], chat_history=formatted_chat_history)  # Render the new template

        return '''
            <form method="POST">
                <input type="text" name="query" placeholder="Enter your query" required>
                <button type="submit">Submit</button>
            </form>
        '''
    finally:
        pass

if __name__ == "__main__":
    app.run(debug=True)

