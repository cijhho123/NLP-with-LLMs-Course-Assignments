# POS Tagging with LLMs: The Hard Parts

## NLP with LLMs - BGU CS - Michael Elhadad - Spring 2025 - HW1

This project explores Part-of-Speech (POS) tagging approaches focusing on the challenging aspects of this fundamental NLP task. By combining traditional machine learning techniques with modern LLM-based approaches, we investigate performance on difficult cases and strategies for improving tagging accuracy.

## Project Overview

The repository implements and evaluates:
- Classical ML approach to POS tagging using Logistic Regression
- LLM-based zero-shot tagging using the Universal Dependencies (UD) tagset
- Various techniques for handling segmentation in real-world text
- Comparative analysis between pipeline and joint segmentation-tagging approaches

## Dataset

We use the [Universal Dependencies English Web Treebank (UD_English-EWT)](https://github.com/UniversalDependencies/UD_English-EWT), a gold standard corpus with syntactic annotations in the CoNLL-U format.

## Repository Structure

```
nlp-with-llms-2025-hw1/
├── ud_pos_tagger_sklearn.ipynb   		# Main notebook with ML implementation and analysis
├── ud_pos_tagger_gemini.py       		# Gemini API implementation for LLM tagging
├── hw1.md                        		# Assignment instructions and requirements
├── error_explanations.pkl        		# Collected LLM explanations for tagging errors
├── generated_hard_sentences.json 		# Synthetically generated challenging sentences
├── llm_tagger_results.pkl        		# Saved results from LLM tagger evaluation
├── tagging_approaches_comparison.pkl 	# Results comparing segmentation approaches
└── README.md                     		# This file
```

## Key Components

### 1. Classical Machine Learning Approach

The Logistic Regression tagger achieves ~95% accuracy using engineered features including:
- Word shape (capitalization, prefixes, suffixes)
- Contextual information (previous/next words)
- Lexical features (the word itself)

### 2. LLM Baseline for POS Tagging

Implementation of a zero-shot LLM tagger with:
- Structured output (JSON) for consistent parsing
- Token-level and sentence-level evaluation metrics
- Comparative error analysis against the LR tagger

### 3. Segmentation Approaches

Four strategies were implemented and compared:
1. **Tokenized baseline**: Using pre-tokenized input (89.74% accuracy)
2. **Original input**: Raw text without specialized segmentation (88.04% accuracy)
3. **Pipeline approach**: Segment first, then tag (87.84% accuracy)
4. **Joint approach**: Segment and tag simultaneously (88.34% accuracy)

### 4. Error Analysis and Improvement

- Identified most challenging POS tag ambiguities (e.g., "that", "as", "to", "like")
- Generated synthetic hard sentences targeting specific error categories
- Used LLM to explain error patterns and improve tagging strategies
- Enhanced the tagger with additional synthetic training data

## Results

1. **Error Analysis**: Function words with high grammatical ambiguity ("that", "as", "to", "like", "out", "for") are the most challenging to tag correctly.

2. **LLM vs. LR Performance**: The LLM-based tagger successfully fixed 40.2% of the errors made by the Logistic Regression model, demonstrating complementary strengths.

3. **Segmentation Impact**: Token segmentation significantly affects tagging performance:
   - Joint segmentation-tagging outperforms the pipeline approach by 0.5%
   - Error propagation is a key issue in pipeline approaches
   - Token alignment is crucial for evaluation when segmentation differs

4. **Synthetic Data**: Adding synthetically generated hard sentences improved the LR tagger accuracy by 0.24% and F1-score by 0.25%.

## Setup and Usage

1. Clone both repositories:
```bash
mkdir hw1
cd hw1
git clone https://github.com/UniversalDependencies/UD_English-EWT.git
git clone https://github.com/melhadad/nlp-with-llms-2025-hw1.git
```

2. Install dependencies:
```bash
cd nlp-with-llms-2025-hw1
pip install -r requirements.txt  # or use uv sync
```

3. Set up API keys for Gemini or Grok:
```bash
# Example for setting up Gemini API
export GOOGLE_API_KEY=your_api_key_here
```

4. Run the notebook:
```bash
jupyter notebook ud_pos_tagger_sklearn.ipynb
```

## Conclusions

The project demonstrates that while traditional ML and modern LLM approaches both perform well, they make different types of errors. LLMs show particular strength in handling semantically ambiguous cases, while statistical approaches benefit from clearer feature definitions.

For text segmentation, the joint approach offers the best balance of performance and practicality, avoiding the error propagation issues present in pipeline methods. The relatively modest differences between approaches (1.4-1.9%) suggest that modern language models are reasonably robust to tokenization variations.

This work highlights the continued importance of studying foundational NLP tasks even in the era of large language models, as understanding the challenges in POS tagging provides insights applicable to more complex downstream tasks.