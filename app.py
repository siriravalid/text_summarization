# app.py

import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Streamlit app
st.title('Text Summarization Tool')
st.write('Input a block of text and get a concise summary.')

# Text input
text_input = st.text_area("Enter text here:", height=300)

# Button to trigger summarization
if st.button('Summarize'):
    if text_input:
        st.write("Generating summary...")

        # Tokenize and encode the input text
        inputs = tokenizer.encode("summarize: " + text_input, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")
