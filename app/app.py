import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import wikipediaapi

# Setup Wikipedia API with user-agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='TruthLensApp/1.0 (contact: your-email@example.com)'  # update email if needed
)

def get_wikipedia_summary(claim):
    page = wiki_wiki.page(claim)
    if page.exists() and len(page.summary.strip()) > 100:
        return page.summary
    else:
        return "No relevant Wikipedia evidence found."

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

# Move model to device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# Label map
label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

# Classification function
def classify_with_evidence(claim):
    claim = claim.strip()
    evidence = get_wikipedia_summary(claim)
    input_text = f"Claim: {claim}\nEvidence: {evidence}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_class = torch.argmax(outputs.logits, dim=1).item()

    return {
        "prediction": label_map.get(pred_class, "UNKNOWN"),
        "evidence": evidence
    }

# User input
user_input = st.text_area("Enter a claim to classify:", height=100)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a claim before classifying.")
    else:
        result = classify_with_evidence(user_input)
        st.markdown(f"### 🧠 Prediction: **{result['prediction']}**")
        st.markdown("### 📚 Retrieved Wikipedia Evidence:")
        st.markdown(result["evidence"])

