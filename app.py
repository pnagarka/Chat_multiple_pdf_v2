import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import csv
import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf(pdf_folder):
    text = ""
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings  = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    vector_stores = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_stores.save_local('faiss_index')

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])
    chain = load_qa_chain(model,chain_type='stuff',prompt=prompt)
    return chain


store_data=[]

def save_to_csv(question, answer, csv_folder='responses'):
    store_data.append({'Question': question, 'Answer': answer})

    # Create a folder if it doesn't exist
    os.makedirs(csv_folder, exist_ok=True)

    # Generate the current date and time as a string
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create a CSV file with the current date and time as the name
    csv_file_path = os.path.join(csv_folder, f'responses_{current_datetime}.csv')

    # Check if the CSV file exists
    is_new_file = not os.path.exists(csv_file_path)

    # Write the question and answer to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(['Question', 'Answer'])  # Write header only if the file is new
        writer.writerow([question, answer])



def user_input(user_question, pdf_folder):
    # Use the new get_pdf function to read PDFs from the specified folder
    raw_text = get_pdf(pdf_folder)

    # Process the text and generate embeddings
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Perform similarity search and get a response from the conversational chain
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {'input_documents': docs, 'question': user_question},
        return_only_outputs=True
    )

    # Save the question and answer to the CSV file
    save_to_csv(user_question, response['output_text'])

    # Display the response in the Streamlit app
    st.write('Reply', response['output_text'])

def main():
    st.set_page_config("Chat with multiple PDF")
    st.header("Chat with PDF using Gemini💁")

    pdf_folder = "path/to/your/pdf/folder"  # Update this with the path to your PDF folder

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        # Use the new get_pdf function to read PDFs from the specified folder
        raw_text = get_pdf(pdf_folder)

        # Process the text and generate embeddings
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Perform similarity search and get a response from the conversational chain
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        new_db = FAISS.load_local('faiss_index', embeddings)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain(
            {'input_documents': docs, 'question': user_question},
            return_only_outputs=True
        )

        # Save the question and answer to the CSV file
        save_to_csv(user_question, response['output_text'])

        # Display the response in the Streamlit app
        st.write('Reply:', response['output_text'])


def save_to_csv_on_close():
    if store_data:
        os.makedirs('responses', exist_ok=True)
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_file_path = os.path.join('responses', f'responses_{current_datetime}.csv')

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Question', 'Answer'])
            for data in store_data:
                writer.writerow([data['Question'], data['Answer']])

    # No need for the file uploader and processing button in this version


