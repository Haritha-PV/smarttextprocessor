import streamlit as st
from io import BytesIO
from fpdf import FPDF
from PyPDF2 import PdfReader
import docx
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

# NLP Task Functions
def entity_recognition(text):
    doc = nlp(text)
    entities = {"Characters": [], "Locations": [], "Dates": [], "Organizations": []}
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["Characters"].append(ent.text)
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            entities["Locations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["Dates"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["Organizations"].append(ent.text)
    
    # Removing duplicates from each category by converting to set and back to list
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    # Formatting the output for better readability
    output = ""
    
    if entities["Characters"]:
        output += "Characters:\n" + ", ".join(entities["Characters"]) + "\n\n"
    if entities["Locations"]:
        output += "Locations:\n" + ", ".join(entities["Locations"]) + "\n\n"
    if entities["Dates"]:
        output += "Dates:\n" + ", ".join(entities["Dates"]) + "\n\n"
    if entities["Organizations"]:
        output += "Organizations:\n" + ", ".join(entities["Organizations"]) + "\n"
    
    return output


def dependency_parsing(text):
    doc = nlp(text)
    parsed_sentences = []
    
    # Enumerate through sentences and add sentence numbers
    for idx, sent in enumerate(doc.sents, 1):  # Start numbering from 1
        parsed_sentences.append(f"{idx}. {str(sent)}")
    
    return "\n".join(parsed_sentences)


def summarize_text(text):
    # Using TextBlob for simple summarization
    blob = TextBlob(text)
    sentences = blob.sentences
    summary = "\n".join(str(sentence) for sentence in sentences[:6])  # Summary with first 6 sentences
    return summary

import re
def topic_modeling(text):
    try:
        if not text:
            return "No text provided for topic modeling."

        # Clean the input text to remove certain characters or redundant terms
        cleaned_text = re.sub(r'\b(?:mother|novel|old|jenny)\b', '', text.lower())
        
        # Vectorize the cleaned text
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([cleaned_text])

        # Perform LDA for topic modeling
        lda = LatentDirichletAllocation(n_components=1, random_state=42)
        lda.fit(X)

        # Extract terms associated with the topic
        terms = vectorizer.get_feature_names_out()
        topic = [terms[i] for i in lda.components_[0].argsort()[:-11:-1]]

        # Remove redundant or irrelevant terms (like character names)
        filtered_topic = list(set(topic))  # Remove duplicates by converting to set

        # Sort the themes alphabetically or by frequency, if needed
        filtered_topic.sort()

        # Return the extracted themes
        return "\n".join(filtered_topic)

    except ValueError as e:
        return f"Error during topic modeling: {e}"

def sentiment_analysis(text):
    # Initialize the TextBlob object
    blob = TextBlob(text)
    
    # Split the text into sentences
    sentences = blob.sentences
    
    # Initialize lists for positive, negative, and neutral sentences
    positive_sentences = []
    negative_sentences = []
    neutral_sentences = []
    
    # Analyze sentiment for each sentence
    for sentence in sentences:
        sentiment = sentence.sentiment.polarity
        
        # Categorize sentences based on sentiment polarity
        if sentiment > 0:
            positive_sentences.append(str(sentence))
        elif sentiment < 0:
            negative_sentences.append(str(sentence))
        else:
            neutral_sentences.append(str(sentence))
    
    # Determine the overall sentiment
    overall_sentiment = "Neutral"
    if len(positive_sentences) > len(negative_sentences):
        overall_sentiment = "Positive"
    elif len(negative_sentences) > len(positive_sentences):
        overall_sentiment = "Negative"
    
    # Start constructing the result string
    result = "Sentiment Analysis:\n"
    
    # Add positive sentences to the result
    if positive_sentences:
        result += "\n**Positive Sentences:**\n"
        for i, sentence in enumerate(positive_sentences, 1):
            result += f"{i}. {sentence}\n"
    else:
        result += "\n**Positive Sentences:** None"
    
    # Add negative sentences to the result
    if negative_sentences:
        result += "\n**Negative Sentences:**\n"
        for i, sentence in enumerate(negative_sentences, 1):
            result += f"{i}. {sentence}\n"
    else:
        result += "\n**Negative Sentences:** None"
    
    # Add neutral sentences to the result
    if neutral_sentences:
        result += "\n**Neutral Sentences:**\n"
        for i, sentence in enumerate(neutral_sentences, 1):
            result += f"{i}. {sentence}\n"
    else:
        result += "\n**Neutral Sentences:** None"
    
    # Add the overall sentiment to the result
    result += f"\n\n**Overall Sentiment:** {overall_sentiment}"
    
    return result


def plot_relationships(text):
    # Process the text
    doc = nlp(text)
    
    # Dictionary to store relationships
    relationships = {}

    # Analyze sentences to extract relationships
    for sent in doc.sents:
        entities = [ent for ent in sent.ents if ent.label_ == "PERSON"]
        if len(entities) >= 2:  # Process sentences with at least two characters
            relation = sent.root.text  # Use the main verb of the sentence
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i + 1:]:
                    # Add to relationships dictionary
                    if entity1.text not in relationships:
                        relationships[entity1.text] = []
                    relationships[entity1.text].append(
                        f"- **{entity2.text}**: {entity1.text} {relation} {entity2.text.lower()} in the context of the story."
                    )

    # Generate formatted output
    formatted_output = []
    for character, relations in relationships.items():
        formatted_output.append(f"### {character}")
        formatted_output.extend(relations)
        formatted_output.append("")  # Add an empty line between characters
    
    return "\n".join(formatted_output)



# Function to create a PDF from task results
def create_pdf(text, entity_recognition_result=None, dependency_parsing_result=None, 
              summarization_result=None, topic_modeling_result=None, 
              sentiment_analysis_result=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Task Results", ln=True, align="C")

     # Call the functions and store their results
    entity_recognition_result = entity_recognition(text) 
    dependency_parsing_result = dependency_parsing(text)  
    summarization_result = summarize_text(text)  
    topic_modeling_result = topic_modeling(text)  
    sentiment_analysis_result = sentiment_analysis(text) 

    # Add results to the PDF
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Entity Recognition:", ln=True)
    pdf.multi_cell(0, 10, str(entity_recognition_result)) 

    pdf.ln(10)
    pdf.cell(200, 10, txt="Dependency Parsing:", ln=True)
    pdf.multi_cell(0, 10, str(dependency_parsing_result))

    pdf.ln(10)
    pdf.cell(200, 10, txt="Summarization:", ln=True)
    pdf.multi_cell(0, 10, str(summarization_result))

    pdf.ln(10)
    pdf.cell(200, 10, txt="Topic Modeling:", ln=True)
    pdf.multi_cell(0, 10, str(topic_modeling_result))

    pdf.ln(10)
    pdf.cell(200, 10, txt="Sentiment Analysis:", ln=True)
    pdf.multi_cell(0, 10, str(sentiment_analysis_result))

    # Generate and return the PDF content
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes
 

# Streamlit App Layout
st.set_page_config(page_title="Script Analyzer", page_icon="📄", layout="wide")

# Add an image to the top of the homepage
st.image("txt8.png", use_container_width=True)  # Replace with your image path

# Layout
st.markdown('<h1 style="color:red; text-align:center; text-transform:uppercase;">SCRIPT ANALYZER 📝</h1>', unsafe_allow_html=True)
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
if "results" not in st.session_state:
    st.session_state.results = {}
file = st.file_uploader("📂 Upload a file (PDF or DOC)", type=["pdf", "doc"])
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

        # New Button to View the File Content
        if st.button("View File Content 📄"):
            st.text_area("File Content", text, height=300)
          
        if st.button("Entity Recognition 🧠🔍"):
            st.session_state.results["Entity Recognition"] = entity_recognition(text)
            st.success(st.session_state.results["Entity Recognition"])
            st.balloons()

        if st.button("Dependency Parsing 🔗📚"):
            st.session_state.results["Dependency Parsing"] = dependency_parsing(text)
            st.success(st.session_state.results["Dependency Parsing"])
            st.balloons()

        if st.button("Summarization 📝✂️"):
            st.session_state.results["Summarization"] = summarize_text(text)
            st.write("Summarization Result:", st.session_state.results["Summarization"])
            st.balloons()

        if st.button("Topic Modeling 🧩🗣️"):
           st.session_state.results["Topic Modeling"] = topic_modeling(text)
          # st.write("Topic Modeling Result:", st.session_state.results["Topic Modeling"])
           st.success(st.session_state.results["Topic Modeling"])
           st.balloons()

        if st.button("Sentiment Analysis 😃😡😐"):
            st.session_state.results["Sentiment Analysis"] = sentiment_analysis(text)
            st.success(st.session_state.results["Sentiment Analysis"])
            st.balloons()

        if st.button("Character/Plot Relationship Analysis 🎭💬"):
            st.session_state.results["Character/Plot Relationship Analysis"] =plot_relationships(text)
            st.success(st.session_state.results["Character/Plot Relationship Analysis"])
            st.balloons()
        

        st.write("Task Results Section")
        

        if st.button("Download Results"):
    # Ensure that results are not empty
           entity_recognition_result = st.session_state.results.get("Entity Recognition", "")
           dependency_parsing_result = st.session_state.results.get("Dependency Parsing", "")
           summarization_result = st.session_state.results.get("Summarization", "")
           topic_modeling_result = st.session_state.results.get("Topic Modeling", "")
           sentiment_analysis_result = st.session_state.results.get("Sentiment Analysis", "")
           character_relationships_result = st.session_state.results.get("Character/Plot Relationship Analysis", "")

    # Check if any result is non-empty
           if any([entity_recognition_result, dependency_parsing_result, summarization_result,
                topic_modeling_result, sentiment_analysis_result, character_relationships_result]):
        
                pdf_content = create_pdf(entity_recognition_result, dependency_parsing_result, summarization_result, 
                                 topic_modeling_result, sentiment_analysis_result, character_relationships_result)



                st.download_button(
                    label="Download Results as PDF",
                    data=pdf_content,
                    file_name="Task_Results.pdf",
                    key="download_pdf_button",
                    mime="application/pdf"
                )
           else:
                st.warning("No results to download.")

# Footer 
st.markdown("---")
st.write("🎉 **Built with Streamlit** | © Your Cool App 2025")