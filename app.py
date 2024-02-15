import streamlit as st
import os
import pandas as pd
import PyPDF2
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter
from io import StringIO
import math

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_lg")


# Function to read resumes from the folder one by one
def pdf_extract(uploaded_file):
    with uploaded_file as f:
        pdf_bytes = f.read()

    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to create candidate profile
def create_profile(text, keyword_dict):
    text = text.lower()
    matcher = PhraseMatcher(nlp.vocab)

    for category, keywords in keyword_dict.items():
        category_keywords = [nlp(keyword.lower()) for keyword in keywords]
        matcher.add(category, None, *category_keywords)

    doc = nlp(text)
    matches = matcher(doc)

    d = []
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        d.append((rule_id, span.text))

    keywords = "\n".join(f'{i[0]} {i[1]}' for i in Counter(d).items())
    df = pd.read_csv(StringIO(keywords), names=['Category', 'Keyword'])

    return df


# Main function to run the app
def main():
    st.title("Resume Screening App")

    # Load keywords from CSV
    keyword_dict = {
        'Statistics': [],
        'NLP': [],
        'Machine Learning': [],
        'Deep Learning': [],
        'R Language': [],
        'Python Language': [],
        'Data Engineering': [],
        'Web Development': []
    }
    keywords_df = pd.read_csv("NLP.csv")
    for category, keywords in keyword_dict.items():
        keywords.extend(keywords_df[category].dropna().tolist())

    # File uploader
    uploaded_file = st.file_uploader("Upload a resume (PDF)", type="pdf")
    if uploaded_file:
        st.write("### Resume Content:")
        text = pdf_extract(uploaded_file)
        st.write(text)

        # Create candidate profile
        profile_df = create_profile(text, keyword_dict)
        st.write("### Candidate Profile:")
        st.write(profile_df)

        # Calculate score
        score = calculate_score(profile_df)
        st.write("### Candidate Score:")
        st.write(score)


# Function to calculate score
def calculate_score(profile_df):
    data_map = {
        'Statistics': 8,
        'NLP': 10,
        'Machine Learning': 9,
        'Deep Learning': 10,
        'R Language': 5,
        'Python Language': 5,
        'Data Engineering': 4,
        'Web Development': 2
    }
    score = 0
    for idx, row in profile_df.iterrows():
        category = row['Category']
        count = row['Count']
        score += data_map.get(category, 0) * count
    return score


if __name__ == '__main__':
    main()
