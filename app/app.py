import streamlit as st
import os
import requests
import zipfile
from io import BytesIO
import time
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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
        self.tokenizers = {}
        self.models = {}
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
        
        # Flag to use fallback if custom models fail
        self.use_fallback = False
        self.fallback_models = {}

    def download_model(self, model_name, url, target_dir):
        """Download model from release URL"""
        try:
            if not os.path.exists(os.path.join(target_dir, "config.json")):
                response = requests.get(url)
                response.raise_for_status()
                
                with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(os.path.dirname(target_dir))
                
                return True, f"✅ {model_name} model downloaded successfully"
            return True, f"✅ {model_name} model already downloaded"
        except Exception as e:
            return False, f"❌ Error downloading {model_name} model: {str(e)}"
    
    def load_custom_model(self, model_name, model_dir):
        """Load a custom model from local directory"""
        try:
            # First try loading with pipeline (the simplest approach)
            try:
                classifier = pipeline(
                    "text-classification",
                    model=model_dir,
                    tokenizer=model_dir
                )
                self.models[model_name] = classifier
                return True, f"✅ {model_name} model loaded successfully with pipeline"
            except Exception as e1:
                st.write(f"Pipeline loading failed for {model_name}: {str(e1)}")
                
                # Second try with direct model loading
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    
                    self.tokenizers[model_name] = tokenizer
                    self.models[model_name] = model
                    return True, f"✅ {model_name} model loaded successfully with direct loading"
                except Exception as e2:
                    st.write(f"Direct model loading failed for {model_name}: {str(e2)}")
                    return False, f"❌ Failed to load {model_name} model"
        except Exception as e:
            return False, f"❌ Error loading {model_name}: {str(e)}"
    
    def load_fallback_models(self):
        """Load fallback models if custom models fail"""
        try:
            # Load pre-trained models for sentiment/entailment as fallbacks
            self.fallback_models["sentiment"] = pipeline("sentiment-analysis")
            self.fallback_models["nli"] = pipeline("zero-shot-classification", 
                                                  model="facebook/bart-large-mnli")
            return True, "✅ Fallback models loaded successfully"
        except Exception as e:
            return False, f"❌ Error loading fallback models: {str(e)}"
    
    def load_models(self):
        """Download and load models from GitHub releases"""
        results = []
        success_count = 0
        
        for model_name, release_url in MODEL_RELEASES.items():
            model_dir = f"models/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Step 1: Download the model
            download_success, download_msg = self.download_model(model_name, release_url, model_dir)
            results.append(download_msg)
            
            # Step 2: If download successful, try to load the model
            if download_success:
                load_success, load_msg = self.load_custom_model(model_name, model_dir)
                results.append(load_msg)
                if load_success:
                    success_count += 1
        
        # If no models were loaded successfully, try fallback
        if success_count == 0:
            self.use_fallback = True
            fallback_success, fallback_msg = self.load_fallback_models()
            results.append(fallback_msg)
            if fallback_success:
                success_count += 1
        
        return success_count > 0, results
    
    def predict_with_pipeline(self, model, claim):
        """Make prediction with pipeline"""
        result = model(claim)
        if isinstance(result, list):
            return result[0]
        return result
    
    def predict_with_model(self, model_name, claim):
        """Make prediction with model and tokenizer"""
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        inputs = tokenizer(claim, return_tensors="pt", truncation=True, padding=True)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get prediction
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        return {"label": pred_idx, "score": confidence}
    
    def predict_with_fallback(self, claim):
        """Make prediction with fallback models"""
        # Use sentiment analysis as rough approximation
        sentiment = self.fallback_models["sentiment"](claim)
        
        # Use zero-shot classification for FEVER-style labels
        hypothesis_template = "This statement is {}."
        fever_result = self.fallback_models["nli"](
            claim, 
            candidate_labels=["true", "false", "uncertain"],
            hypothesis_template=hypothesis_template
        )
        
        # Map results to our format
        if fever_result["labels"][0] == "true":
            fever_pred = "SUPPORTS"
            liar_pred = "true"
        elif fever_result["labels"][0] == "false":
            fever_pred = "REFUTES"
            liar_pred = "false"
        else:
            fever_pred = "NOT ENOUGH INFO"
            liar_pred = "half-true"
        
        # Return unified result
        return {
            "prediction": fever_pred,
            "confidence": fever_result["scores"][0],
            "nuanced_prediction": liar_pred,
            "model_results": {
                "sentiment": {
                    "prediction": sentiment[0]["label"],
                    "confidence": sentiment[0]["score"],
                    "inference_time_ms": 0
                },
                "zero-shot": {
                    "prediction": fever_pred,
                    "confidence": fever_result["scores"][0],
                    "inference_time_ms": 0
                }
            }
        }
    
    def predict(self, claim, weights=None, return_details=False):
        """Make prediction using the ensemble"""
        # Use fallback if needed
        if self.use_fallback:
            return self.predict_with_fallback(claim)
            
        # Use default weights if none provided
        if weights is None:
            weights = self.default_weights
        
        # Store results for each model
        model_results = {}
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                start_time = time.time()
                
                # Check if it's a pipeline or a model+tokenizer
                if isinstance(model, pipeline) or callable(getattr(model, "__call__", None)):
                    # It's a pipeline
                    result = self.predict_with_pipeline(model, claim)
                    # Extract label and confidence
                    if isinstance(result, dict):
                        pred_idx = int(result["label"].split("_")[-1]) if "_" in result["label"] else 0
                        confidence = result["score"]
                    else:
                        pred_idx = 0
                        confidence = 0.5
                else:
                    # It's a model with tokenizer
                    result = self.predict_with_model(model_name, claim)
                    pred_idx = result["label"]
                    confidence = result["score"]
                
                # Process based on model type
                if model_name == "deberta":  # LIAR framework
                    label = self.liar_classes[pred_idx] if pred_idx < len(self.liar_classes) else self.liar_classes[0]
                    framework = "LIAR"
                    cross_framework_idx = self.liar_to_fever[pred_idx] if pred_idx < len(self.liar_classes) else 1
                    cross_framework_label = self.fever_classes[cross_framework_idx]
                else:  # FEVER framework
                    label = self.fever_classes[pred_idx] if pred_idx < len(self.fever_classes) else self.fever_classes[1]
                    framework = "FEVER"
                    cross_framework_idx = self.fever_to_liar[pred_idx] if pred_idx < len(self.fever_classes) else 1
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
                st.write(f"Error in {model_name} prediction: {str(e)}")
        
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
    st.session_state.models_loading = False
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm TruthLens, your AI fact-checking assistant. Ask me to verify any claim or statement, and I'll analyze its truthfulness using my ensemble of models trained on fact-checking datasets. How can I help you today?"}
    ]

# Sidebar model loading
with st.sidebar:
    st.title("TruthLens Models")
    if not st.session_state.models_loaded and not st.session_state.models_loading:
        if st.button("Load Models", type="primary"):
            st.session_state.models_loading = True
            with st.spinner("Downloading and loading models... This may take a minute."):
                success, results = st.session_state.ensemble.load_models()
                for result in results:
                    st.write(result)
                
                if success:
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.error("❌ Failed to load models. Using fallback mode.")
                    st.session_state.models_loaded = True  # Use fallback
            
            st.session_state.models_loading = False
    elif st.session_state.models_loading:
        st.info("⏳ Loading models... Please wait.")
    else:
        if st.session_state.ensemble.use_fallback:
            st.warning("⚠️ Using fallback mode due to model loading issues.")
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
