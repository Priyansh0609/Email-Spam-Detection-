import streamlit as st
import nltk
import pickle

import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('Mnb_model.pkl', 'rb'))

st.title("Email Spam Detection")

input_sms = st.text_input("Enter your message")
if st.button("Predict"):

    #1 preprocess
    transformed_text = transform_text(input_sms)

    # 2) Vectorizer

    vector_input = tfidf.transform([transformed_text])

    # 3) Predict
    result = model.predict(vector_input)

    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

