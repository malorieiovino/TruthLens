import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 1. Load your model & tokenizer (assumes you have a directory with saved model weights)
tokenizer = DistilBertTokenizer.from_pretrained("fever_distilbert_model")
model = DistilBertForSequenceClassification.from_pretrained("fever_distilbert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

st.title("FEVER Claim Classification")

user_input = st.text_area("Enter a claim to classify:")

if st.button("Classify"):
    with torch.no_grad():
        # Tokenize & move to device
        inputs = tokenizer(
            user_input, return_tensors='pt', truncation=True, padding=True
        ).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        label = label_map[pred]
        st.write(f"**Prediction:** {label}")

