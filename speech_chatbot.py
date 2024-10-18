import streamlit as st
import speech_recognition as sr  # For speech recognition
import nltk  # For natural language processing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Load the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize necessary components
recognizer = sr.Recognizer()  # Initialize the recognizer for speech recognition
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load and preprocess the chatbot corpus (e.g., a book text)
def load_text():
    try:
        # Example text file path
        file_path = r'C:/Users/HARDEY/Documents/GOMYCODE/Deep Learning/alice_in_wonderland.txt'
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\n', ' ')
    except FileNotFoundError:
        st.error("Text file not found.")
        return ""

# Preprocessing function for the chatbot text
def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return [lemmatizer.lemmatize(token) for token in tokens]

# Tokenize and preprocess the corpus
text = load_text()
sentences = nltk.sent_tokenize(text)
corpus = [preprocess(sentence) for sentence in sentences]

# Function to transcribe speech into text
def transcribe_speech():
    try:
        with sr.Microphone() as source:
            st.info("Listening... Please speak.")
            audio = recognizer.listen(source)
            st.info("Processing...")
            # Use Google Speech Recognition to transcribe speech into text
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return None

# Chatbot response function using Jaccard similarity
def get_response(query):
    query_processed = preprocess(query)
    max_similarity = 0
    best_response = "I'm sorry, I couldn't find an appropriate response."
    
    for sentence, original_sentence in zip(corpus, sentences):
        similarity = jaccard_similarity(query_processed, sentence)
        if similarity > max_similarity:
            max_similarity = similarity
            best_response = original_sentence
    return best_response

# Jaccard similarity function
def jaccard_similarity(query_set, sentence_set):
    intersection = len(set(query_set).intersection(set(sentence_set)))
    union = len(set(query_set).union(set(sentence_set)))
    return intersection / union

# Streamlit app for text and speech chatbot
def app():
    st.title("Speech-Enabled Chatbot")

    # Display chatbot instructions
    st.markdown("""
    ### Instructions:
    1. Type your question in the text box and click "Submit" to get a response.
    2. Or click "Use Speech" to provide input via your microphone. The speech will be transcribed and processed.
    """)

    # Option 1: Text input
    user_input_text = st.text_input("Enter your message (Text):")
    if st.button("Submit"):
        if user_input_text:
            response = get_response(user_input_text)
            st.write(f"Chatbot: {response}")
        else:
            st.warning("Please enter a message.")

    # Option 2: Speech input
    if st.button("Use Speech"):
        st.info("Click the 'Allow' button in your browser if asked for microphone access.")
        transcribed_text = transcribe_speech()
        if transcribed_text:
            st.write(f"You said: {transcribed_text}")
            response = get_response(transcribed_text)
            st.write(f"Chatbot: {response}")

# Run the Streamlit app
if __name__ == "__main__":
    app()
