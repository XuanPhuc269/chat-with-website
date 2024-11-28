import streamlit as st
import sqlite3
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

def get_vectorstore_from_url(url):
    try:
        # Try to access the vector store
        loader = WebBaseLoader(url)
        document = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)

        # Create vector store from documents
        vector_store = Chroma.from_documents(
            document_chunks, OpenAIEmbeddings(), persist_directory="path_to_your_database"
        )

    except sqlite3.OperationalError as e:
        # If the table doesn't exist (OperationalError), recreate the vector store
        print(f"Error: {e}. Reinitializing vector store...")
        vector_store = Chroma.from_documents(
            document_chunks, OpenAIEmbeddings(), persist_directory="path_to_your_database"
        )

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relavant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt) 

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
     # Create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    # If no relevant information is found, return the default response
    if not response or 'answer' not in response or not response['answer'].strip():
        return "I don't know. The information you request is not available in the provided resource."

    return response['answer']


# App configc
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")    

if website_url is None or website_url == "":
    st.info("Please enter a website URL.")

else:
    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)    




