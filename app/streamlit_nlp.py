import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Set title
st.title("🔍 FEVER Claim Classification")

# Load tokenizer and model from local files
tokenizer = DistilBertTokenizer.from_pretrained(
    "models/distilbert_base_model",
    local_files_only=True
)

model = DistilBertForSequenceClassification.from_pretrained(
    "models/distilbert_fine_tuned_fever",
    local_files_only=True
)

model.eval()

# Define labels
label_map = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

# User input
user_input = st.text_area("Enter a claim to classify:", height=100)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a claim before classifying.")
    else:
        # Tokenize
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Show prediction
        st.markdown(f"### 🧠 Prediction: **{label_map[predicted_class]}**")

