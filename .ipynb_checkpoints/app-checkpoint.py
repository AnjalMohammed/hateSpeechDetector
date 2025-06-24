import streamlit as st
from transformers import pipeline

# Load model
@st.cache_resource  # ensures it's loaded only once
def load_model():
    return pipeline("text-classification", model="unitary/toxic-bert")

classifier = load_model()

# UI
st.title("Hate Speech Detector")
st.write("This app uses a pre-trained BERT model to detect toxic content in text.")

user_input = st.text_area("Enter a tweet or text here:")

if st.button("Classify"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            prediction = classifier(user_input)[0]
            label = prediction["label"]
            score = prediction["score"]

            st.markdown(f"**Label:** {label}")
            st.markdown(f"**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter some text to classify.")