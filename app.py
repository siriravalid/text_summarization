
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
st.title('Text Summarization Tool')
st.write('Input a block of text and get a concise summary.')
text_input = st.text_area("Enter text here:", height=300)
if st.button('Summarize'):
    if text_input:
        st.write("Generating summary...")
        inputs = tokenizer.encode("summarize: " + text_input, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")
