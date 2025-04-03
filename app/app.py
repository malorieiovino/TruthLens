import streamlit as st
import os
import requests
import zipfile
from io import BytesIO
import time
from transformers import pipeline, AutoTokenizer

# Page configuration
st.set_page_config(
    page_title="TruthLens: Fact-Checking Chatbot",
    page_icon="🔍",
    layout="wide"
)

# GitHub release URLs - update these with your actual release URLs
MODEL_RELEASES = {
    "deberta": "https://github.com/malorieiovino/TruthLens/releases/download/v1.0/deberta_liar.zip",
    "distilbert": "https://github.com/malorieiovino/TruthLens/releases/download/v1.0/distilbert_fever.zip",
    "roberta": "https://github.com/malorieiovino/TruthLens/releases/download/v1.0/roberta_fever.zip"
}

# Create models directory
os.makedirs("models", exist_ok=True)

# Helper class for the ensemble
class FactCheckingEnsemble:
    def __init__(self):
        self.pipelines = {}
        self.device = -1  # Use CPU
        
        # Class mappings
        self.liar_classes = ["pants-on-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
        self.fever_classes = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        
        # Cross-framework mappings
        self.liar_to_fever = {
            0: 1, 1: 1, 2: 1,  # Lower truthfulness -> REFUTES
            3: 2,              # Half-true -> NOT ENOUGH INFO
            4: 0, 5: 0         # Higher truthfulness -> SUPPORTS
        }
        
        self.fever_to_liar = {
            0: 5,  # SUPPORTS -> true
            1: 1,  # REFUTES -> false
            2: 3   # NOT ENOUGH INFO -> half-true
        }
        
        # Default weights for ensemble
        self.default_weights = {
            "deberta": 1.0,
            "distilbert": 1.0,
            "roberta": 1.0
        }
    
    def load_models(self):
        """Download and load models from GitHub releases"""
        for model_name, release_url in MODEL_RELEASES.items():
            model_dir = f"models/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Only download if not already present
            if not os.path.exists(f"{model_dir}/config.json"):
                try:
                    response = requests.get(release_url)
                    response.raise_for_status()
                    
                    # Extract the zip file
                    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                        zip_ref.extractall("models")
                    
                    st.success(f"✅ {model_name} model downloaded successfully")
                except Exception as e:
                    st.error(f"❌ Error downloading {model_name} model: {str(e)}")
                    continue
            
            # Load model as pipeline
            try:
                self.pipelines[model_name] = pipeline(
                    "text-classification", 
                    model=model_dir,
                    tokenizer=model_dir,
                    device=self.device
                )
                st.success(f"✅ {model_name} model loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading {model_name}: {str(e)}")
    
    def predict(self, claim, weights=None, return_details=False):
        """Make prediction using the ensemble"""
        if not self.pipelines:
            st.error("No models available. Please make sure at least one model is loaded.")
            return None
            
        # Use default weights if none provided
        if weights is None:
            weights = self.default_weights
        
        # Store results for each model
        model_results = {}
        
        # Make predictions with each model
        for model_name, pipe in self.pipelines.items():
            try:
                start_time = time.time()
                
                # Get prediction from pipeline
                result = pipe(claim, top_k=None)
                
                # Process based on model type
                if model_name == "deberta":  # LIAR framework
                    # Map label ID to label text
                    label_id = int(result[0]['label'].split('_')[-1])
                    label = self.liar_classes[label_id]
                    confidence = result[0]['score']
                    framework = "LIAR"
                    cross_framework_idx = self.liar_to_fever[label_id]
                    cross_framework_label = self.fever_classes[cross_framework_idx]
                else:  # FEVER framework
                    # Map label ID to label text
                    label_id = int(result[0]['label'].split('_')[-1])
                    label = self.fever_classes[label_id]
                    confidence = result[0]['score']
                    framework = "FEVER"
                    cross_framework_idx = self.fever_to_liar[label_id]
                    cross_framework_label = self.liar_classes[cross_framework_idx]
                
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Store results
                model_results[model_name] = {
                    "prediction": label,
                    "confidence": confidence,
                    "framework": framework,
                    "cross_framework_prediction": cross_framework_label,
                    "inference_time_ms": inference_time
                }
            except Exception as e:
                st.error(f"Error in {model_name} prediction: {str(e)}")
        
        if not model_results:
            return None
            
        # Calculate ensemble result in FEVER framework (simpler for user understanding)
        fever_votes = {label: 0.0 for label in self.fever_classes}
        
        for model_name, result in model_results.items():
            if result["framework"] == "FEVER":
                fever_votes[result["prediction"]] += result["confidence"] * weights.get(model_name, 1.0)
            else:
                fever_votes[result["cross_framework_prediction"]] += result["confidence"] * weights.get(model_name, 1.0)
        
        # Determine winning prediction
        ensemble_prediction = max(fever_votes, key=fever_votes.get)
        total_weight = sum(weights.get(model, 1.0) for model in model_results.keys())
        ensemble_confidence = fever_votes[ensemble_prediction] / total_weight if total_weight > 0 else 0
        
        # Map back to LIAR for more nuanced output
        ensemble_liar_idx = self.fever_to_liar[self.fever_classes.index(ensemble_prediction)]
        ensemble_liar_prediction = self.liar_classes[ensemble_liar_idx]
        
        if return_details:
            return {
                "prediction": ensemble_prediction,
                "confidence": ensemble_confidence,
                "nuanced_prediction": ensemble_liar_prediction,
                "fever_votes": fever_votes,
                "model_results": model_results
            }
        else:
            return ensemble_prediction, ensemble_confidence

# Function to extract claim from user message
def extract_claim(message):
    prefixes = ["fact-check:", "verify:", "is it true that", "check if"]
    
    for prefix in prefixes:
        if prefix in message.lower():
            return message.lower().split(prefix, 1)[1].strip().rstrip("?")
    
    # If no prefix is found, return the whole message
    return message

# Function to format chatbot response
def format_fact_check_response(claim, result):
    # Determine verdict and emoji
    if result["prediction"] == "SUPPORTS":
        verdict = "likely true"
        emoji = "✅"
        explanation = "The evidence supports this claim."
    elif result["prediction"] == "REFUTES":
        verdict = "likely false"
        emoji = "❌"
        explanation = "The evidence contradicts this claim."
    else:
        verdict = "inconclusive"
        emoji = "⚠️"
        explanation = "There isn't enough information to verify this claim."
    
    # Build response
    response = f"{emoji} Based on my analysis, this claim is **{verdict}** (confidence: {result['confidence']*100:.1f}%).\n\n"
    response += f"{explanation}\n\n"
    response += f"**Nuanced assessment:** {result['nuanced_prediction'].replace('-', ' ').title()}\n\n"
    
    # Add model details in a collapsible section
    response += "<details><summary>See detailed analysis</summary>\n\n"
    
    for model_name, model_result in result["model_results"].items():
        response += f"**{model_name}**: {model_result['prediction']} (confidence: {model_result['confidence']*100:.1f}%)\n\n"
    
    response += "</details>"
    
    return response

# Initialize the app
st.title("TruthLens: AI Fact-Checking Chatbot")
st.markdown("""
Welcome to TruthLens, an AI-powered fact-checking assistant that helps verify claims using an ensemble of advanced language models.
Ask me to fact-check any claim by typing it below or starting your message with "fact-check:" or "verify:".
""")

# Initialize session state
if 'ensemble' not in st.session_state:
    st.session_state.ensemble = FactCheckingEnsemble()
    st.session_state.models_loaded = False
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm TruthLens, your AI fact-checking assistant. Ask me to verify any claim or statement, and I'll analyze its truthfulness using my ensemble of models trained on fact-checking datasets. How can I help you today?"}
    ]

# Sidebar model loading
with st.sidebar:
    st.title("TruthLens Models")
    if not st.session_state.models_loaded:
        if st.button("Load Models", type="primary"):
            with st.spinner("Downloading and loading models... This may take a minute."):
                st.session_state.ensemble.load_models()
                if st.session_state.ensemble.pipelines:
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.error("❌ Failed to load any models.")
    else:
        st.success("✅ Models loaded and ready to use!")
    
    st.markdown("---")
    st.markdown("""
    ### About TruthLens
    This chatbot uses an ensemble of transformer models:

    * **DeBERTa**: Fine-tuned on LIAR dataset (6-class)
    * **DistilBERT**: Fine-tuned on FEVER dataset (3-class)
    * **RoBERTa**: Fine-tuned on FEVER dataset (3-class)

    The system combines multiple fact-checking perspectives to provide more nuanced assessments.
    
    Created for NLP Assessment 1, 2024-25.
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me to fact-check something...", disabled=not st.session_state.models_loaded):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        # Extract the claim from user message
        claim = extract_claim(prompt)
        
        # Check if we need to analyze a claim
        is_fact_check = any(prefix in prompt.lower() for prefix in ["fact-check", "verify", "is it true", "check if"]) or len(st.session_state.messages) <= 2
        
        if is_fact_check and st.session_state.models_loaded:
            # It's a fact-checking request
            with st.spinner("Analyzing claim..."):
                result = st.session_state.ensemble.predict(claim, return_details=True)
            
            if result:
                response = format_fact_check_response(claim, result)
                st.markdown(response)
            else:
                st.markdown("I'm sorry, I couldn't analyze this claim. There might be an issue with the models or the input format.")
        else:
            # It's a general chat message
            if not st.session_state.models_loaded:
                response = "I'd be happy to help fact-check that, but I need my fact-checking models to be loaded first. Please click the 'Load Models' button in the sidebar to get started."
            else:
                response = "I'm a fact-checking assistant. To verify a claim, please phrase your request as 'Fact-check: [your claim]' or 'Verify: [your claim]'."
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Example fact-checking prompts
with st.expander("Example fact-checking prompts"):
    examples = [
        "Fact-check: Building a wall on the U.S.-Mexico border will take literally years.",
        "Verify: Wisconsin is on pace to double the number of layoffs this year.",
        "Is it true that the US has the highest inflation rate in the world?",
        "Fact-check: Climate change is just a theory with no scientific consensus.",
        "Check if vaccines have been scientifically proven to cause autism."
    ]
    
    columns = st.columns(2)
    for i, example in enumerate(examples):
        with columns[i % 2]:
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                # This will insert the example into the chat input when clicked
                st.session_state.example_input = example
                st.rerun()

# Handle example selection
if 'example_input' in st.session_state:
    example = st.session_state.example_input
    del st.session_state.example_input
    # The chat_input can't be directly set, so we use messages as a workaround
    st.session_state.messages.append({"role": "user", "content": example})
    st.rerun()
