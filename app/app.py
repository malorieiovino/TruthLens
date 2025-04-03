import streamlit as st
import io
import requests
import numpy as np
from typing import Dict, List, Optional
import torch
from transformers import pipeline

class TruthLensClassifier:
    def __init__(self):
        # GitHub raw content base URL
        self.base_url = "https://raw.githubusercontent.com/malorieiovino/TruthLens/main/models/"
        
        # Model configuration
        self.model_configs = [
            {
                'name': 'deberta',
                'url': self.base_url + 'deberta_liar_model.pkl',
                'labels': ['mostly_true', 'half_true', 'barely_true', 'false', 'mostly_false', 'pants_on_fire']
            },
            {
                'name': 'distilbert',
                'url': self.base_url + 'distilbert_fever_model.pkl',
                'labels': ['true', 'false', 'not_enough_info']
            },
            {
                'name': 'roberta',
                'url': self.base_url + 'roberta_fever_model.pkl',
                'labels': ['true', 'false', 'not_enough_info']
            }
        ]
        
        # Initialize models
        self.models = {}
        self._load_models()
        
        # Predefined corrections for specific claims
        self.fact_corrections = {
            "black people are dangerous": {
                "verdict": "False",
                "explanation": "This claim is a harmful and racist stereotype with no scientific or factual basis. Crime rates are influenced by complex socioeconomic factors, not race. Generalizing criminal behavior to an entire racial group is not only scientifically incorrect but deeply unethical.",
                "model_details": {}
            }
        }
    
    def _load_models(self):
        """
        Load models from GitHub raw content
        """
        for model_config in self.model_configs:
            try:
                # Download model file
                response = requests.get(model_config['url'])
                response.raise_for_status()
                
                # Create model pipeline
                # Note: This is a placeholder and may need adjustment based on actual model format
                model = pipeline('text-classification', 
                                 model=io.BytesIO(response.content),
                                 labels=model_config['labels'])
                
                self.models[model_config['name']] = model
            except Exception as e:
                st.error(f"Failed to load {model_config['name']} model: {e}")
                self.models[model_config['name']] = None
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Analyze claim using multiple models
        """
        # Check for predefined corrections first
        normalized_text = text.lower().strip()
        for key, correction in self.fact_corrections.items():
            if key in normalized_text:
                # Add model-specific details
                model_details = {}
                for model_name, model in self.models.items():
                    if model:
                        try:
                            model_result = model(text)[0]
                            model_details[model_name] = {
                                'label': model_result['label'],
                                'score': model_result['score']
                            }
                        except Exception as e:
                            model_details[model_name] = f"Error: {e}"
                
                correction['model_details'] = model_details
                return correction
        
        # Perform multi-model analysis
        model_results = {}
        for model_name, model in self.models.items():
            if model:
                try:
                    result = model(text)[0]
                    model_results[model_name] = {
                        'label': result['label'],
                        'score': result['score']
                    }
                except Exception as e:
                    model_results[model_name] = f"Error: {e}"
        
        # Aggregate results
        return {
            "verdict": "Needs Verification",
            "explanation": "This claim requires further investigation across multiple models.",
            "model_details": model_results
        }

def main():
    st.set_page_config(
        page_title="TruthLens: Political Fact-Checker",
        page_icon="🕵️",
        layout="wide"
    )
    
    st.title("🕵️ TruthLens: Unfiltered Political Fact-Checking")
    
    # Sidebar model status
    st.sidebar.title("Model Status")
    
    # Initialize the classifier
    classifier = TruthLensClassifier()
    
    # Display model loading status
    for model_name, model in classifier.models.items():
        if model:
            st.sidebar.success(f"{model_name.capitalize()} Model: ✅ Loaded")
        else:
            st.sidebar.error(f"{model_name.capitalize()} Model: ❌ Failed to Load")
    
    # Input text area
    input_text = st.text_area(
        "Enter claim to fact-check:", 
        height=200, 
        placeholder="Paste the political statement or claim you want to verify..."
    )
    
    # Fact-check button
    if st.button("Verify Claim"):
        if not input_text:
            st.warning("Please enter a claim to analyze.")
            return
        
        # Perform fact-checking
        result = classifier.predict(input_text)
        
        # Result display
        st.header("Fact-Checking Result")
        
        # Verdict display
        st.markdown(f"**Verdict:** {result.get('verdict', 'Inconclusive')}")
        
        # Explanation
        st.write(result.get('explanation', 'No additional context available.'))
        
        # Detailed Model Results
        st.subheader("Model Analysis")
        model_details = result.get('model_details', {})
        
        for model_name, details in model_details.items():
            st.markdown(f"**{model_name.capitalize()} Model:**")
            if isinstance(details, dict):
                st.write(f"Label: {details.get('label', 'N/A')}")
                st.write(f"Confidence: {details.get('score', 0)*100:.2f}%")
            else:
                st.error(details)

if __name__ == "__main__":
    main()
