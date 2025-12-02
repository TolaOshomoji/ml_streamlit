# nlp_app.py

# PART 1 (35 points): NLP App Creation
# Create an NLP application that will predict whether the user's text input has a positive, negative, or neutral sentiment

# 1. Create a directory for the application nlp and python script nlp_app.py. Import the following libraries.

# Streamlit Library
import streamlit as st

# Other libraries
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob


# 2. Create the GUI that takes user input for sentiment analysis. Provide a title, and define a text area (the input text can span multiple lines). Create a button "Analyze") that will then return the sentiment polarity and sentiment for the given text. [HINT: st.title(), st.subheader(), st.text_area(), st.button()]

st.title("Tola's Sentiment Analysis App")
st.subheader("Enter some text and I’ll classify it as positive, negative, or neutral")

# Multi-line text input from the user
text = st.text_area("Type your text here:", height=200)

# 3. Process the cleaned text when the Analyze button is clicked.
# A. Use the following code to clean the text (remove whitespaces, web links, punctuations, and digits) and apply lemmatization to the text that will give a single word for the inflected forms.

def clean_text(text: str) -> str:
    # Keeping only letters and digits
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # Removes "'s"
    text = re.sub(r"\'s", " ", text)
    # Removing links if any
    text = re.sub(r"http\S+", " link ", text)
    # Removes punctuations and numbers
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)

    # Splitting Text
    text = text.split()

    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)

    return text


# B. Use the TextBlob() function to predict the sentiment of the cleaned text and compute the sentiment polarity to determine if the sentiment in positive, negative or neutral. [HINT: blob = TextBlob(text)  sentiment_score = blob.sentiment.polarity ]

# C. Display different alerts st.success for positive, st.warning for negative and st.info for neutral sentiments with a display text and emojis as shown below. 

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text before clicking Analyze.")
    else:
        # Clean and lemmatize the text
        cleaned_text = clean_text(text)

        # Use TextBlob to compute sentiment polarity
        blob = TextBlob(cleaned_text)
        sentiment_score = blob.sentiment.polarity  # in [-1.0, 1.0]
        result = sentiment_score

        # Display different alerts based on the sign of the polarity
        if result > 0:
            custom_emoji = ':blush:'
            st.success('Happy : {}'.format(custom_emoji))
        elif result < 0:
            custom_emoji = ':disappointed:'
            st.warning('Sad : {}'.format(custom_emoji))
        else:
            custom_emoji = ':confused:'
            st.info('Confused : {}'.format(custom_emoji))

        # Show the numeric polarity score
        st.success("Polarity Score is: {}".format(result))
