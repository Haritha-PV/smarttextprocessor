import streamlit as st
from io import BytesIO
from fpdf import FPDF
from PyPDF2 import PdfReader
import docx

# Function to call OpenAI API (dummy placeholder for now)
def call_openai(prompt):
    return f"Processed output for: {prompt}"

# NLP Task Functions
def entity_recognition(text):
    prompt = f"Extract characters, locations, dates, and organizations from the following text:\n{text}"
    return call_openai(prompt)

def dependency_parsing(text):
    prompt = f"Break down the following sentence into simpler sentences:\n{text}"
    return call_openai(prompt)

def summarize_text(text):
    prompt = f"Summarize the following text in 3-5 sentences:\n{text[:4000]}"
    return call_openai(prompt)

def topic_modeling(text):
    prompt = f"Identify the main themes or topics in the following text:\n{text}"
    return call_openai(prompt)

def sentiment_analysis(text):
    prompt = f"Analyze the sentiment of the following text as positive, negative, or neutral:\n{text}"
    return call_openai(prompt)

def plot_relationships(text):
    prompt = f"For each character, describe their relationships with other characters.\n{text}"
    return call_openai(prompt)

# Helper functions
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# PDF download function
def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for task, result in results.items():
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(0, 10, f"{task}:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, result)
        pdf.ln()

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Streamlit App Layout
st.set_page_config(page_title="Smart Text Processor", page_icon="📄", layout="wide")

# Add an image to the top of the homepage
st.image("txt8.png", use_container_width=True) 

# Layout
st.markdown('<h1 style="color:red; text-align:center; text-transform:uppercase;">Smart Text Processor 📝</h1>', unsafe_allow_html=True)
st.write("Perform text analysis with ease and style. Choose your task below! 👇")

# Sidebar
st.sidebar.markdown('<h1 style="color:red;">📄✨ INTERACTIVE TEXT MANAGER 🖋️</h1>', unsafe_allow_html=True)
st.sidebar.write("Make text processing fun and easy! 🚀")
# Insert image in the sidebar
st.sidebar.image("gif.gif", use_container_width=True)  # Replace with your image path
st.sidebar.info("""
🚀 **What can you do here?**

🔍 Extract entities, break down sentences, summarize, analyze sentiment, and more.  
📂 Upload PDF or DOC files and process text effortlessly.  
💾 Download results for future use.  
""")


# File upload
file = st.file_uploader("📂 Upload a file (PDF or DOC)", type=["pdf", "doc"])
st.info("Supported formats: PDF, DOC.")
if file:
    # Extract text based on file type
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "application/msword" or file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file)
    else:
        text = None
        st.error("Unsupported file format!")

    if text:
        st.success("File uploaded and text extracted successfully!")

        # Task buttons and results
        results = {}
        if st.button("Entity Recognition 🧠🔍"):
            results["Entity Recognition"] = entity_recognition(text)
            st.success(results["Entity Recognition"])
            st.balloons()

        if st.button("Dependency Parsing 🔗📚"):
            results["Dependency Parsing"] = dependency_parsing(text)
            st.success(results["Dependency Parsing"])
            st.balloons()
            
        if st.button("Summarization 📝✂️"):
            results["Summarization"] = summarize_text(text)
            st.success(results["Summarization"])
            st.balloons()

        if st.button("Topic Modeling 🧩🗣️"):
            results["Topic Modeling"] = topic_modeling(text)
            st.success(results["Topic Modeling"])
            st.balloons()

        if st.button("Sentiment Analysis 😃😡😐"):
            results["Sentiment Analysis"] = sentiment_analysis(text)
            st.success(results["Sentiment Analysis"])
            st.balloons()

        if st.button("Character/Plot Relationship Analysis 🎭💬"):
           results["Character/Plot Relationship Analysis"] = plot_relationships(text)
           st.success(results["Character/Plot Relationship Analysis"])
           st.balloons()

        if results and st.button("Download Results as PDF 💾📥"):
           pdf_output = create_pdf(results)
           st.download_button(
              label="Download PDF 💾📥",
              data=pdf_output.getvalue(),
              file_name="results.pdf",
              mime="application/pdf"
           )
           st.balloons()
# Footer
st.markdown("---")
st.write("🎉 **Built with Streamlit** | © Your Cool App 2025")