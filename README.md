# TruthLens: AI-Powered Fact-Checking Chatbot

## Overview

TruthLens is a fact-checking chatbot that uses an ensemble of transformer models to evaluate the truthfulness of claims. The system combines multiple fact-checking perspectives by integrating models trained on different frameworks:

- **DeBERTa**: Fine-tuned on the LIAR dataset (6-class truthfulness scale)
- **DistilBERT**: Fine-tuned on the FEVER dataset (3-class verification system)
- **RoBERTa**: Fine-tuned on the FEVER dataset (3-class verification system)

By merging these complementary frameworks, TruthLens provides both nuanced truthfulness assessment and decisive verification.

## Features

- **Cross-Framework Integration**: Combines LIAR's 6-class truthfulness scale with FEVER's 3-class verification system
- **Ensemble Prediction**: Weighted voting system across multiple models for improved accuracy
- **Conversational Interface**: Natural chat interface for easy interaction
- **Explainable Results**: Transparent analysis showing which models contributed to the verdict
- **Bias Detection**: Sensitivity to potential biases in fact-checking
- **Optimized Performance**: Models fine-tuned for efficiency and accuracy

## Demo

TruthLens is deployed and available at: [https://truthlens-gdyc5cjscpwyv74rgkjbou.streamlit.app](https://truthlens-gdyc5cjscpwyv74rgkjbou.streamlit.app)

## Technical Details

### Model Architecture

TruthLens employs a weighted ensemble of three transformer models:

1. **DeBERTa**: Provides nuanced truthfulness classification with 6 classes:
   - pants-on-fire
   - false
   - barely-true
   - half-true
   - mostly-true
   - true

2. **DistilBERT & RoBERTa**: Provide decisive verification with 3 classes:
   - SUPPORTS
   - REFUTES
   - NOT ENOUGH INFO

### Cross-Framework Mapping

The system maps between frameworks using carefully designed correspondence:

| LIAR (6-class) | FEVER (3-class) |
|----------------|-----------------|
| pants-on-fire  | REFUTES         |
| false          | REFUTES         |
| barely-true    | REFUTES         |
| half-true      | NOT ENOUGH INFO |
| mostly-true    | SUPPORTS        |
| true           | SUPPORTS        |

### Optimization Techniques

The models have been optimized using:
- FP16 precision (half-precision)
- Model pruning (20% sparsity)
- Inference pipeline optimization

## Development and Training

### Datasets

- **LIAR**: Contains 12.8K human-labeled short statements from PolitiFact, annotated with fine-grained truthfulness ratings
- **FEVER**: Contains 185K claims generated from Wikipedia, manually verified by humans

### Training Process

1. Initial model development with baseline classifiers
2. Advanced model development with transformers
3. Fine-tuning pre-trained models for domain adaptation
4. Evaluation across multiple metrics
5. Implementation of explainability techniques
6. Optimization for deployment

## Using TruthLens

1. Visit the [TruthLens App](https://truthlens-gdyc5cjscpwyv74rgkjbou.streamlit.app)
2. Click "Load Models" in the sidebar
3. Ask the chatbot to fact-check a claim using formats like:
   - "Fact-check: [your claim]"
   - "Verify: [your claim]"
   - "Is it true that [your claim]?"

## Future Improvements

- Integration with live news sources
- Support for multi-lingual fact-checking
- Enhanced bias detection capabilities
- Source citation for verification

## License

For educational purposes only.

## Acknowledgments

- LIAR dataset from William Yang Wang
- FEVER dataset from the FEVER workshop
- Hugging Face for transformer implementations
- Streamlit for deployment platform

