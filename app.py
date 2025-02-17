import streamlit as st
from streamlit_chat import message as chat_message
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

def main():
    # Page Configuration
    st.set_page_config(page_title="AI-Powered Chatbot", layout="wide")
    st.title("📚 AI-Powered Chatbot")
    st.markdown("---")

    # Sidebar for input options
    with st.sidebar:
        st.header("📥 Input Options")
        input_type = st.radio("Select Input Type", options=["Web URL", "PDF Upload"])

        if input_type == "Web URL":
            url = st.text_input("Enter Web URL")
            if st.button("Add URL"):
                process_url(url)

        elif input_type == "PDF Upload":
            pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if pdf_file:
                process_pdf(pdf_file)

    st.markdown("---")

    # Chat Layout
    st.subheader("💬 Chat Conversation")
    chat_container = st.container()  # Scrollable chat container
    with chat_container:
        st.markdown(
            """
            <style>
                .scrollable-container {
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 5px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        chat_container.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

        # Display chat conversation
        conversation = st.session_state.get('conversation', [])
        for idx, (q, a) in enumerate(conversation):
            chat_message(q, is_user=True, key=f"q_{idx}")
            chat_message(a, is_user=False, key=f"a_{idx}")

        chat_container.markdown('</div>', unsafe_allow_html=True)

    # Chat Input
    st.markdown("---")
    query = st.text_input("Enter your question", key="query_input", label_visibility="collapsed")
    send_button = st.button("Send", key="send_button")

    if send_button and query:
        if 'loaded_data' not in st.session_state:
            st.warning("Please upload a PDF or enter a URL to proceed.")
        else:
            conversation = process_query(query, conversation)
            st.session_state['conversation'] = conversation
            st.experimental_rerun()  # Refresh to display the new conversation

    # Download Options
    if conversation:
        st.markdown("---")
        st.subheader("📄 Download Options")
        questions = [f"Q{idx + 1}: {q}" for idx, (q, _) in enumerate(conversation)]
        selected_question = st.selectbox("Select a Question to Download", questions)

        if st.button("Download Specific Answer as PDF"):
            selected_index = questions.index(selected_question)
            q, a = conversation[selected_index]
            pdf_data = generate_specific_pdf(q, a)
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="specific_answer.pdf",
                mime="application/pdf",
            )

        if st.button("Download Entire Conversation as PDF"):
            pdf_data = generate_full_pdf(conversation)
            st.download_button(
                label="Download Entire Conversation",
                data=pdf_data,
                file_name="entire_conversation.pdf",
                mime="application/pdf",
            )

    # Reset Chat
    if st.button("Reset Chat"):
        reset_bot()
        st.experimental_rerun()

def process_url(url):
    """Process the URL and prepare the data."""
    if not url:
        st.warning("Please enter a valid URL.")
        return
    try:
        loader = RecursiveUrlLoader(url=url, max_depth=3, extractor=lambda x: Soup(x, "html.parser").text)
        data = loader.load()
        process_data(data)
        st.success(f"Successfully processed URL: {url}")
    except Exception as e:
        st.error(f"Error processing URL: {e}")

def process_pdf(pdf_file):
    """Process the uploaded PDF and prepare the data."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        process_data(text)
        st.success(f"Successfully processed PDF: {pdf_file.name}")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

def process_data(data):
    """Process the text data and store embeddings."""
    try:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=15000, chunk_overlap=500)
        docs = text_splitter.create_documents([data])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_documents(docs, embeddings).as_retriever()

        st.session_state['loaded_data'] = (docs, vector_store)
    except Exception as e:
        st.error(f"Error processing data: {e}")

def process_query(query, conversation):
    """Process the user query."""
    docs, vector_store = st.session_state['loaded_data']

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, convert_system_message_to_human=True)
    memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        llm=model,
        max_token_limit=2400,
    )

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=model, retriever=vector_store, memory=memory, verbose=False
    )
    response = chatbot({"question": query})
    conversation.append((query, response["answer"]))
    return conversation

def reset_bot():
    """Reset the bot to its initial state."""
    st.session_state.clear()

def generate_specific_pdf(question, answer):
    """Generate a PDF file for a specific question and answer."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Chatbot Reply")
    y -= 40

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColorRGB(0.1, 0.5, 0.8)
    pdf.drawString(50, y, f"User: {question}")
    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.setFillColorRGB(0, 0, 0)
    for line in split_text(answer, width - 100, pdf):
        pdf.drawString(50, y, line)
        y -= 20

        if y < 50:
            pdf.showPage()
            y = height - 40

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

def generate_full_pdf(conversation):
    """Generate a PDF for the entire conversation."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Full Chat Conversation")
    y -= 40

    for idx, (q, a) in enumerate(conversation):
        pdf.setFont("Helvetica-Bold", 12)
        pdf.setFillColorRGB(0.1, 0.5, 0.8)
        pdf.drawString(50, y, f"User: {q}")
        y -= 30

        pdf.setFont("Helvetica", 12)
        pdf.setFillColorRGB(0, 0, 0)
        for line in split_text(a, width - 100, pdf):
            pdf.drawString(50, y, line)
            y -= 20

            if y < 50:
                pdf.showPage()
                y = height - 40

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

def split_text(text, max_width, pdf):
    """Split text into lines."""
    words = text.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        if pdf.stringWidth(test_line, "Helvetica", 12) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    return lines

if __name__ == "__main__":
    main()
