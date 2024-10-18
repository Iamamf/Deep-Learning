
import streamlit as st  # For web interface
import speech_recognition as sr  # For speech recognition
import os  # For file saving
import time  # For pause/resume feature
from pydub import AudioSegment  # For audio file handling

# List of available APIs for speech recognition
apis = {
    "Google Web Speech API": "google",
    "Sphinx (Offline)": "sphinx",
}

# List of languages supported by the Google Web Speech API
languages = {
    "English": "en-US",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "German": "de-DE"
}

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to transcribe speech from audio input
def transcribe_speech(source, api_choice, language_code):
    try:
        with source as audio_source:
            recognizer.adjust_for_ambient_noise(audio_source)
            audio = recognizer.listen(audio_source)  # Capture the audio
            st.info("Recognizing...")

        # Select API based on user's choice
        if api_choice == "google":
            return recognizer.recognize_google(audio, language=language_code)  # Recognize speech in the chosen language
        elif api_choice == "sphinx":
            return recognizer.recognize_sphinx(audio)  # Sphinx works offline, but doesn't support language selection
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from {api_choice} API; {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return None

# Function to save transcribed text to a file
def save_to_file(transcribed_text):
    file_name = f"transcription_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    with open(file_name, 'w') as file:
        file.write(transcribed_text)
    st.success(f"Transcription saved as {file_name}")

# Function to handle pause and resume functionality
def pause_resume(recognition_state):
    if recognition_state["paused"]:
        st.info("Resuming speech recognition...")
        recognition_state["paused"] = False
    else:
        st.warning("Speech recognition paused.")
        recognition_state["paused"] = True

# Streamlit app interface
def app():
    st.title("Enhanced Speech Recognition App")

    # Step 1: Select speech recognition API
    st.subheader("1. Select Speech Recognition API")
    api_choice = st.selectbox("Choose an API:", list(apis.keys()))

    # Step 2: Select language for recognition (if applicable)
    st.subheader("2. Select Language")
    language_choice = st.selectbox("Choose the language you will speak in:", list(languages.keys()))

    # Step 3: Start speech recognition process
    st.subheader("3. Start Speech Recognition")
    recognition_state = {"paused": False}  # Store the paused state
    if st.button("Start Speech Recognition"):
        st.info("Speak into your microphone...")
        
        # Step 4: Pause/Resume feature
        if st.button("Pause/Resume"):
            pause_resume(recognition_state)
        
        # Audio input from microphone
        if not recognition_state["paused"]:
            mic = sr.Microphone()

            with mic as source:
                st.info("Listening...")

                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)

                # Listen and transcribe using the selected API and language
                transcribed_text = transcribe_speech(source, apis[api_choice], languages[language_choice])

                # Display the transcribed text
                if transcribed_text:
                    st.write(f"Transcribed Text: {transcribed_text}")
                    
                    # Step 5: Save the transcribed text to a file
                    if st.button("Save Transcription to File"):
                        save_to_file(transcribed_text)

# Run the Streamlit app
if __name__ == "__main__":
    app()
