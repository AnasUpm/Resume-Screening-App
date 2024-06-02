# importing all required libraries
import math;
import PyPDF2
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import spacy
#spacy.cli.download("en_core_web_sm")
import en_core_web_lg;
import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")
from spacy.matcher import PhraseMatcher

# Function to read resumes from the folder one by one
mypath = r'D:\downloads\archive\Resume_EY_Diverse'
onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
print(os.listdir(mypath))


def pdfextract(file):
    fileReader = PyPDF2.PdfReader(open(file, 'rb'))
    countpage = len(fileReader.pages)
    count = 0
    text = []
    while count < countpage:
        pageObj = fileReader.pages[count]
        count += 1
        t = pageObj.extract_text()
        print(t)
        text.append(t)
    return text


# function to read resume ends


# function that does phrase matching and builds a candidate profile
def create_profile(file, idx):
    text = pdfextract(file)
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    # below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('C:\\Users\Acer\PycharmProjects\ResumeScreening\\NLP.csv')
    stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis=0)]
    NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis=0)]
    ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis=0)]
    DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis=0)]
    R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis=0)]
    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis=0)]
    Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis=0)]
    WebDev_words = [nlp(text) for text in keyword_dict['Web Development'].dropna(axis=0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats_words)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('ML', None, *ML_words)
    matcher.add('DL', None, *DL_words)
    matcher.add('R', None, *R_words)
    matcher.add('Python', None, *python_words)
    matcher.add('DE', None, *Data_Engineering_words)
    matcher.add('WebDev', None, *WebDev_words)
    doc = nlp(text)

    d = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start: end]
        d.append((rule_id, span.text))
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i, j in Counter(d).items())

    df = pd.read_csv(StringIO(keywords), names=['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ', n=1).tolist(), columns=['Subject', 'Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(', n=1).tolist(), columns=['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'], df2['Keyword'], df2['Count']], axis=1)
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]

    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2), names=['Candidate Name'])

    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis=1)
    dataf['Candidate Name'] = dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0])

    ## Calculating Scroing of Candidate :

    tot_sum = df3['Count'].sum();
    score.append((dataf['Candidate Name'][0], tot_sum));

    return (dataf)


# DE 4 DL 10 ML 9 NLP 10 Python 5 R 5 Stats 8 WebDev 2

# Assuming we are shortlisting for AI profile.
data_map = dict();


def data_dict():
    global data_map;

    data_map["DE"] = 4;
    data_map["DL"] = 10;
    data_map["ML"] = 9;
    data_map["NLP"] = 10;
    data_map["Python"] = 5;
    data_map["R"] = 5;
    data_map["Stats"] = 8;
    data_map["WebDev"] = 2;


def score_calc():
    global score, final_database2, data_map;
    mm = []
    col = [];
    for i in final_database2.columns:
        col.append(i);

    name = "";
    for i in range(len(final_database2)):
        name = final_database2.iloc[i, 0];
        subsum = 0;
        for j in range(1, len(final_database2.columns)):
            num1 = final_database2.iloc[i, j];
            num2 = data_map.get(col[j]);
            subsum += int(num1) * num2;

        mm.append((name, subsum));

    score = mm;


# Tuple Sorting Function
def Sort_Tuple(tup):
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):

        for j in range(0, lst - i - 1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup


from prettytable import PrettyTable


def Sel_printer(show):
    global score, selected;
    num_sel = math.ceil(len(score) / show);
    selected = [];

    Sort_Tuple(score);

    j = len(score) - 1;
    while (j > 0 and num_sel > 0):
        selected.append((score[j][0], score[j][1]));
        num_sel -= 1;
        j -= 1;

    t = PrettyTable(['Name', 'Score'])
    for i in range(len(selected)):
        t.add_row([selected[i][0], selected[i][1]])

    return t


# code to count words under each category and visulaize it through Matplotlib
def data_plot():
    global final_database2;
    final_database2 = final_database['Keyword'].groupby(
        [final_database['Candidate Name'], final_database['Subject']]).count().unstack()
    final_database2.reset_index(inplace=True)
    final_database2.fillna(0, inplace=True)
    new_data = final_database2.iloc[:, 1:]
    new_data.index = final_database2['Candidate Name']
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10})
    ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(25, 7), stacked=True)
    labels = []
    for j in new_data.columns:
        for i in new_data.index:
            label = str(j) + ": " + str(new_data.loc[i][j])
            labels.append(label)
    patches = ax.patches
    for label, rect in zip(labels, patches):
        width = rect.get_width()
        if width > 0:
            x = rect.get_x()
            y = rect.get_y()
            height = rect.get_height()
            ax.text(x + width / 2., y + height / 2., label, ha='center', va='center')
    plt.show()


# function ends

final_database = pd.DataFrame();
final_database2 = pd.DataFrame();
score = [];
selected = [];

def read_resumes_from_folder(folder_path):
    resumes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.pdf'):
            with open(os.path.join(folder_path, filename), 'rb') as file:
                resume_bytes = file.read()
                try:
                    resume_text = resume_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 decoding fails, try decoding with 'latin-1'
                    resume_text = resume_bytes.decode('latin-1')
                cleaned_resume = clean_resume(resume_text)
                resumes.append(cleaned_resume)
    return resumes
def main():
    st.title("Resume Screening App")
    #uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
    global final_database, final_database2, data_map;

    i = 0;
    while i < len(onlyfiles):
        file = onlyfiles[i];
        dat = create_profile(file, i);
        #st.write("dat : ",dat)
        final_database = pd.concat([final_database, dat], ignore_index=True);
        i += 1;
        print(final_database);

    data_dict();

    data_plot();

    print("final database 2 \n",final_database2);

    score_calc();

    print("Score Card of the Candidates : ");
    Sel_printer(1);
    st.write(Sel_printer(1))
    print("Selected Candidates :");
    Sel_printer(10);
    st.write(Sel_printer(10))

    # Update this variable with the path to the folder containing resumes
    resumes_folder_path = r'D:\downloads\archive\Resume_EY_Diverse'

    # Read resumes from the folder
    resumes = read_resumes_from_folder(resumes_folder_path)

    # Vectorize resumes
    input_features = tfidfd.transform(resumes)

    # Predict categories for resumes
    predictions = clf.predict(input_features)

    # Map category IDs to category names
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    # Predicted categories for each resume
    predicted_categories = [category_mapping.get(pred, "Unknown") for pred in predictions]



    # Display predicted categories for each resume
    for i, category in enumerate(predicted_categories):
        st.write(f"Resume {i + 1}: Predicted Category - {category}")


if __name__ == '__main__':
    main();