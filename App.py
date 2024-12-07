import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download('stopwords')

#Loading models
cf = pickle.load(open('cf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

#Importing the function cleanres here
def cleanres(txt):
    cleantxt = re.sub(r'http\S+\s?', '', txt) #To remove links 'https.example.com'
    cleantxt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', cleantxt) #To remove emails
    cleantxt = re.sub(r'\s?#\w+', '', cleantxt) #To remove Hashtags
    cleantxt = re.sub(r'\s?@\w+', '', cleantxt) #To remove
    #cleantxt = re.sub(r'[^A-Za-z0-9\s]', '', cleantxt)
    cleantxt = re.sub(r'[^\w\d\s]', '', cleantxt)
    cleantxt = re.sub(r'\s+', ' ', cleantxt)
    return cleantxt

#Creating the website
def main():
    st.title('Resume Screening App')
    upload_file = st.file_uploader("Upload a PDF", type="pdf")

    if upload_file is not None:
        reader = PdfReader(upload_file)
        res_txt = ''
        for page in reader.pages:
            res_txt+=page.extract_text()
        cleaned_resume = cleanres(res_txt)
        res_trans = tfidf.transform([cleaned_resume])
        prediction_id = cf.predict(res_trans)[0]


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

        category_name = category_mapping.get(prediction_id, 'unknown')
        st.write('The Predicted category of the uploaded resume is: ', category_name)






#Python main
if __name__ == '__main__':
    main()