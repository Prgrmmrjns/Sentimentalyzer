import tensorflow as tf
import streamlit as st
import numpy as np
from translatepy import Translator

def process_sentence(user_input):
    translator = Translator()
    model = tf.keras.models.load_model("token_char_model_2")
    emotions = ['Anger :angry:', 'Fear :fearful:', 'Joy :smile:', 'Love :two_hearts:','Surprise :astonished:']
    result = translator.translate(user_input, "English")
    user_input = result.result
    sentence = np.expand_dims(user_input, axis=0)
    chars = np.expand_dims(" ".join(list(user_input)), axis=0)
    probs = model.predict((sentence, chars))
    preds = preds = np.argsort(-probs)
    preds = tf.squeeze(preds)
    emotion = emotions[preds[0]]
    return emotion

st.title("Sentimentalyzer 	:smile: ")
st.write("Sentimentalyzer uses Natural Language Processing to predict the mood of a sentence.")
st.write("The tool uses a translation APIs and is thus able to translate your sentences into English before NLP.")
st.write("The six sentiments that can be predicted are: anger :angry:, fear :fearful:, joy :smile:, love :two_hearts:, sadness :cry: and surprise :astonished:.")

user_input = st.text_input("Enter your sentence here:")
if st.button("Submit"):
    st.markdown(process_sentence(user_input))