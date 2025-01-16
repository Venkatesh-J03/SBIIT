import streamlit as st
from PyPDF2 import PdfReader
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    """Extract text from the uploaded PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    """Preprocess text by removing punctuation, converting to lowercase, and removing stopwords."""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", "", text)

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]

    return filtered_words

def generate_wordcloud(words):
    """Generate a word cloud from the list of words."""
    wordcloud = WordCloud(width=800, height=400, max_words=500, background_color='white').generate(" ".join(words))
    return wordcloud

def main():
    st.title("PDF WordCloud Generator")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        # Preprocess text
        with st.spinner("Preprocessing text..."):
            words = preprocess_text(text)

        # Generate word cloud
        with st.spinner("Generating word cloud..."):
            wordcloud = generate_wordcloud(words)

        # Display word cloud
        st.subheader("Word Cloud")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Show raw text and word count
        st.subheader("Extracted Text (Preview)")
        st.text(" ".join(words[:500]))  # Display a preview of the first 500 words
        st.write(f"Total words after preprocessing: {len(words)}")

if __name__ == "__main__":
    main()

