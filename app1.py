import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import csv
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    # Ensure the conversation function is initialized in the session state
    if "conversation" not in st.session_state:
        # Initialize it here or ensure it's done in the main setup
        pass 

    # Get the response for the current user question
    response = st.session_state.conversation({'question': user_question})

    # Initialize chat history if not already done
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
        st.session_state.last_displayed_msg_index = -1  # Initialize the index

    # Extend the chat history with the new response
    st.session_state.chat_history.extend(response['chat_history'])

    # Display messages that have not been displayed yet
    for i in range(st.session_state.last_displayed_msg_index + 1, len(st.session_state.chat_history)):
        message = st.session_state.chat_history[i]
        if i % 2 == 0:  # Assuming even indices are user messages, odd indices are bot responses
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    # Update the index of the last displayed message
    st.session_state.last_displayed_msg_index = len(st.session_state.chat_history) - 1



def save_to_csv(question, answer, csv_file_path='responses.csv'):
    csv_file_created = False

    if not csv_file_created:
        responses_directory = 'responses'
        os.makedirs(responses_directory, exist_ok=True)
        csv_file_path = os.path.join(responses_directory, csv_file_path)

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.path.getsize(csv_file_path) == 0:
                writer.writerow(['Question', 'Answer'])  # Write header only if the file is new

        csv_file_created = True

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([question, answer])



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state or "vectorstore" not in st.session_state:
        st.header("Chat with multiple PDFs :books:")
        pdf_directory = "C:/Users/15516/OneDrive/Desktop/GLOBAL_AI/Chat_multiple_pdf_v2/Chat_multiple_pdf_v2/pdf_folder"  # Replace with the path to your PDF directory
        pdf_docs = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
        
        # Process PDFs
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        
        # Only create vectorstore if it doesn't exist
        if "vectorstore" not in st.session_state:
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.vectorstore = vectorstore
        else:
            vectorstore = st.session_state.vectorstore
        
        conversation_chain = get_conversation_chain(vectorstore)
        st.session_state.conversation = conversation_chain
    else:
        st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        # Handle user input
        handle_userinput(user_question)
        save_to_csv(user_question, st.session_state.chat_history)

if __name__ == '__main__':
    main()
