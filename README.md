# Resume Screening App

## Description

The Resume Screening App is a tool designed to assist in the automated screening of resumes by leveraging natural language processing (NLP) techniques. This application uses Streamlit for the user interface and employs the Spacy library for NLP tasks. The app can read PDF resumes, extract relevant text, and identify key skills and qualifications based on predefined keywords. It then generates a candidate profile and calculates a score based on the frequency of the identified skills and their respective importance.

### Features

1. **PDF Resume Upload**: Users can upload resumes in PDF format.
2. **Text Extraction**: Extracts text content from the uploaded PDF resume.
3. **Keyword Matching**: Uses a PhraseMatcher to identify and match relevant keywords in the resume text.
4. **Candidate Profile Generation**: Creates a profile of the candidate by categorizing identified keywords.
5. **Score Calculation**: Computes a candidate score based on the occurrence and importance of keywords.

### How It Works

1. **Load Keywords**: Keywords are loaded from a CSV file and categorized into various domains such as Statistics, NLP, Machine Learning, Deep Learning, R Language, Python Language, Data Engineering, and Web Development.
2. **File Upload**: The user uploads a PDF resume through the file uploader interface.
3. **Text Extraction**: The app reads the content of the uploaded PDF file.
4. **Profile Creation**: The extracted text is processed to identify and categorize keywords, creating a candidate profile.
5. **Score Calculation**: A score is calculated based on the identified keywords and their respective weights.

### Dependencies

- Streamlit
- PyPDF2
- Spacy
- pandas

## Usage

1. Run the app using Streamlit.
2. Upload a PDF resume.
3. View the extracted resume content.
4. Review the generated candidate profile.
5. Check the calculated candidate score.

This app provides a streamlined and efficient way to screen resumes, highlighting key skills and qualifications to aid in the recruitment process.
