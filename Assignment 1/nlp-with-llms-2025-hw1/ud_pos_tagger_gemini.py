# --- Imports ---
import os
import json
from google import genai
#import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum
import time

# 
gemini_model = 'gemini-2.0-flash-lite'

# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    """Universal Dependencies Part-of-Speech Tagset (17 core tags)"""
    ADJ = "ADJ"       # adjective
    ADP = "ADP"       # adposition (preposition, postposition)
    ADV = "ADV"       # adverb
    AUX = "AUX"       # auxiliary verb
    CCONJ = "CCONJ"   # coordinating conjunction
    DET = "DET"       # determiner
    INTJ = "INTJ"     # interjection
    NOUN = "NOUN"     # noun
    NUM = "NUM"       # numeral
    PART = "PART"     # particle
    PRON = "PRON"     # pronoun
    PROPN = "PROPN"   # proper noun
    PUNCT = "PUNCT"   # punctuation
    SCONJ = "SCONJ"   # subordinating conjunction
    SYM = "SYM"       # symbol
    VERB = "VERB"     # verb
    X = "X"           # other

# ========= Define more Pydantic models for structured output =========
# Task 2 - tagger models
class TokenPOS(BaseModel):
    """Represents a token with its part-of-speech tag."""
    token: str = Field(description="The word or token from the text.")
    pos_tag: UDPosTag = Field(description="The Universal Dependencies part-of-speech tag.")

class SentencePOS(BaseModel):
    """Represents a sentence with POS-tagged tokens."""
    tokens: List[TokenPOS] = Field(description="List of tokens with their POS tags.")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

# Task 3 - segment models
class TokenSegment(BaseModel):
    """Model representing a single segmented token."""
    token: str = Field(description="The segmented token")
    start_char: int = Field(description="Starting character position in original text")
    end_char: int = Field(description="Ending character position in original text")

class SegmentedSentence(BaseModel):
    """Model representing a fully segmented sentence."""
    original_text: str = Field(description="The original unsegmented text")
    tokens: List[TokenSegment] = Field(description="List of segmented tokens")



# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "YOUR_API_KEY"
        print("Using secondary key")
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    genai.configure(api_key=api_key)

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# ========== Task 2 - POS tagging ==========
# --- Function to Perform POS Tagging ---
def tag_sentences_ud(text_to_tag: str, token_counts: Optional[List[int]] = None) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API.

    Args:
        text_to_tag: The sentence or text to be tagged.
        token_counts: Optional list of expected token counts for each sentence.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Base prompt
    base_prompt = f"""You are a linguistic expert in part-of-speech tagging.

Task: Perform part-of-speech (POS) tagging on the input text using the Universal Dependencies (UD) tagset.

IMPORTANT ABOUT TOKENIZATION:
- The input text is already pre-tokenized - each space-separated word is EXACTLY ONE token
- Each space-separated item in the input must receive exactly one POS tag
- DO NOT split or merge any tokens - preserve the exact tokenization provided
- DO NOT perform any additional tokenization
- Punctuation marks that appear as separate space-delimited items are separate tokens
- Count each space-separated item as exactly one token

Universal Dependencies POS Tags:
ADJ: adjective
ADP: adposition (preposition, postposition)
ADV: adverb
AUX: auxiliary verb  
CCONJ: coordinating conjunction
DET: determiner
INTJ: interjection
NOUN: noun
NUM: numeral
PART: particle
PRON: pronoun
PROPN: proper noun
PUNCT: punctuation
SCONJ: subordinating conjunction
SYM: symbol
VERB: verb
X: other
"""

    # Add token count enforcement if provided
    if token_counts:
        token_count_instructions = """
CRITICAL TOKEN COUNT REQUIREMENT:
- You MUST tag EXACTLY the number of tokens specified for each sentence
- Each space-separated word in the input is exactly one token 
- Your response MUST contain exactly the same tokens in the same order as the input
- Do not split, merge, add, or remove any tokens
- If a sentence has N tokens specified, you must provide exactly N tags
"""
        base_prompt += token_count_instructions

    # Complete the prompt with the input text
    prompt = base_prompt + f"\nInput text to tag:\n{text_to_tag}"

    # Make the API call
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
        },
    )

    # Parse and return the response
    res: TaggedSentences = response.parsed
    return res


def build_error_explanation_prompt(sentence, tokens, correct_tags, predicted_tags):
    """
    Build a prompt for the LLM to explain POS tagging errors in a sentence.
    """
    error_lines = []
    for w, c, p in zip(tokens, correct_tags, predicted_tags):
        if c != p:
            error_lines.append(f"{w:<20} C: {c:<8} P: {p:<8} **** Error")
        else:
            error_lines.append(f"{w:<20} {c:<8}")

    error_table = "\n".join(error_lines)
    num_errors = sum(1 for c, p in zip(correct_tags, predicted_tags) if c != p)

    prompt = f"""
Here is a sentence with POS tagging errors:

Sentence:
{' '.join(tokens)}

Number of errors: {num_errors}

Token-level tags:
{error_table}

For each error (where the correct tag and predicted tag differ), provide a JSON object with:
- word
- correct_tag
- predicted_tag
- explanation (why the error happened)
- category (short error type)

IMPORTANT: Use these standardized error categories:
1. "Lexical Ambiguity: [CORRECT_TAG]/[PREDICTED_TAG]" - When the word can function as multiple POS
2. "Contextual Misinterpretation: [CORRECT_TAG]/[PREDICTED_TAG]" - When context determines the correct tag
3. "Named Entity Error: [CORRECT_TAG]/[PREDICTED_TAG]" - For proper noun classification issues
4. "Symbol/Punctuation Confusion: [CORRECT_TAG]/[PREDICTED_TAG]" - For SYM vs PUNCT issues
5. "Numeral Classification: [CORRECT_TAG]/[PREDICTED_TAG]" - For issues with numbers/numerals
6. "Capitalization Issue: [CORRECT_TAG]/[PREDICTED_TAG]" - When capitalization affects tagging

Examples:
- For a word that should be ADJ but was tagged as NOUN: "Lexical Ambiguity: ADJ/NOUN"
- For a proper noun tagged as common noun: "Named Entity Error: PROPN/NOUN"

Return a JSON array of explanations.
"""
    return prompt

def extract_json_from_llm_response(llm_response):
    """
    Extract a JSON array from the LLM response string.
    """
    try:
        start = llm_response.find('[')
        end = llm_response.rfind(']')
        if start != -1 and end != -1:
            return json.loads(llm_response[start:end+1])
        else:
            return []
    except Exception as e:
        print("Failed to parse LLM response:", e)
        return []

def collect_llm_error_explanations(hard_sentences):
    """
    Collect explanations generated by the Gemini LLM on errors.
    Args:
        hard_sentences: List of (tagged_sentence, num_errors, predicted_tags)
    Returns:
        List of (sentence, explanations) dicts
    """
    explanations_list = []
    client = genai.Client(api_key=api_key)
    
    for idx, (tagged_sentence, num_errors, predicted_tags) in enumerate(hard_sentences):
        tokens = [w for w, _ in tagged_sentence]
        correct_tags = [t for _, t in tagged_sentence]
        prompt = build_error_explanation_prompt(" ".join(tokens), tokens, correct_tags, predicted_tags)
        
        # Direct Gemini API call
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            
        )
        llm_response = response.text if hasattr(response, "text") else str(response)
        explanations = extract_json_from_llm_response(llm_response)
        explanations_list.append({
            "sentence": " ".join(tokens),
            "explanations": explanations
        })
        #print(f"\nSentence {idx+1}: { ' '.join(tokens) }")
        #print(json.dumps(explanations, indent=2))
    
    return explanations_list


def generate_hard_sentences_api(prompt, max_retries=3, initial_backoff=2):
    """
    Make an API request to Gemini to generate hard sentences.
    
    Args:
        prompt: The prompt to send to the model
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff time in seconds
        
    Returns:
        List of generated sentences or None if failed
    """
    import time
    import json
    
    retry_count = 0
    client = genai.Client(api_key=api_key)
    
    while retry_count <= max_retries:
        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
            )
            
            if response:
                response_text = response.text
                
                # Extract JSON from response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    raise Exception("Could not find JSON array in response")
            else:
                raise Exception("Empty response from API")
                
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                backoff_time = initial_backoff * (2 ** (retry_count - 1))
                print(f"API call failed, retrying in {backoff_time}s... ({retry_count}/{max_retries})")
                time.sleep(backoff_time)
            else:
                print(f"Failed after {max_retries} retries: {str(e)}")
                return None
    
    return None


# ========== Task 3 - segmentation ==========
def segment_sentence_ud(original_sentence, examples=None, max_retries=3, initial_backoff=2):
    """
    Segment a sentence according to UD guidelines using few-shot prompting.
    
    Args:
        original_sentence: The original unsegmented sentence
        examples: Optional list of example segmentations for few-shot learning
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        
    Returns:
        SegmentedSentence object with tokens
    """
    # Prepare system prompt with segmentation instructions
    system_prompt = """
    You are a linguistic expert in text segmentation according to Universal Dependencies (UD) guidelines.
    Your task is to segment text into tokens following these rules:
    
    1. Punctuation marks are separate tokens (periods, commas, parentheses, etc.)
    2. Hyphens in compounds are separate tokens (e.g., "search-engine" → "search - engine")
    3. Possessives ('s) are separate tokens (e.g., "John's" → "John 's")
    4. Contractions are split (e.g., "don't" → "do n't", "I'll" → "I 'll")
    5. Multi-word expressions remain as separate tokens
    
    For each token, identify its start and end character positions in the original text.
    """
    
    # Prepare user prompt
    user_prompt = f"Please segment this sentence according to UD guidelines:\n\n{original_sentence}"
    
    # Add few-shot examples if provided
    if examples:
        example_text = "\nHere are some examples of proper segmentation:\n\n"
        for ex in examples:
            example_text += f"Original: {ex['original']}\n"
            example_text += f"Segmented: {ex['segmented']}\n\n"
        user_prompt = example_text + user_prompt
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Retry logic
    retry_count = 0
    while retry_count <= max_retries:
        try:
            # Make API call
            response = client.models.generate_content(
                model=gemini_model,
                contents=system_prompt + "\n\n" + user_prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': SegmentedSentence,
                }
            )
            
            # Check if we got a parsed response
            if hasattr(response, "parsed"):
                return response.parsed
            else:
                raise Exception("Failed to parse response")
                
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                backoff_time = initial_backoff * (2 ** (retry_count - 1))
                print(f"Retrying in {backoff_time}s... ({retry_count}/{max_retries})")
                time.sleep(backoff_time)
            else:
                print(f"Failed after {max_retries} retries: {str(e)}")
                return None
    
    return None

def pipeline_segmentation_and_tagging(original_sentence, examples=None):
    """
    Two-step pipeline: segment first, then tag.
    
    Args:
        original_sentence: Original unsegmented sentence
        examples: Optional list of example segmentations for few-shot learning
        
    Returns:
        TaggedSentences object
    """
    # Step 1: Segment
    segmentation_result = segment_sentence_ud(original_sentence, examples)
    if not segmentation_result or not hasattr(segmentation_result, "tokens"):
        return None
    
    # Extract tokens as a space-separated string
    segmented_text = " ".join([t.token for t in segmentation_result.tokens])
    
    # Step 2: Tag
    tagging_result = tag_sentences_ud(segmented_text)
    
    return tagging_result

def joint_segmentation_and_tagging(original_sentence, examples=None, max_retries=3, initial_backoff=2):
    """
    Joint approach: segment and tag in a single step.
    
    Args:
        original_sentence: Original unsegmented sentence
        examples: Optional list of example segmentations for few-shot learning
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        
    Returns:
        TaggedSentences object
    """
    # Prepare system prompt
    system_prompt = """
    You are a linguistic expert in text processing according to Universal Dependencies (UD) guidelines.
    Your task is to:
    
    1. First segment the text into tokens following these rules:
       - Punctuation marks are separate tokens (periods, commas, parentheses, etc.)
       - Hyphens in compounds are separate tokens (e.g., "search-engine" → "search - engine")
       - Possessives ('s) are separate tokens (e.g., "John's" → "John 's")
       - Contractions are split (e.g., "don't" → "do n't", "I'll" → "I 'll")
    
    2. Then assign the correct Universal Dependencies POS tag to each token:
       ADJ: adjective
       ADP: adposition (preposition, postposition)
       ADV: adverb
       AUX: auxiliary verb
       CCONJ: coordinating conjunction
       DET: determiner
       INTJ: interjection
       NOUN: noun
       NUM: numeral
       PART: particle
       PRON: pronoun
       PROPN: proper noun
       PUNCT: punctuation
       SCONJ: subordinating conjunction
       SYM: symbol
       VERB: verb
       X: other
    """
    
    # Prepare user prompt
    user_prompt = f"Please segment and tag this sentence according to UD guidelines:\n\n{original_sentence}"
    
    # Add few-shot examples if provided
    if examples:
        example_text = "\nHere are some examples of proper segmentation and tagging:\n\n"
        for ex in examples:
            example_text += f"Original: {ex['original']}\n"
            if 'segmented_tagged' in ex:
                example_text += f"Segmented and tagged: {ex['segmented_tagged']}\n\n"
            else:
                example_text += f"Segmented: {ex['segmented']}\n\n"
        user_prompt = example_text + user_prompt
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Retry logic
    retry_count = 0
    while retry_count <= max_retries:
        try:
            # Make API call
            response = client.models.generate_content(
                model=gemini_model,
                contents=system_prompt + "\n\n" + user_prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': TaggedSentences,
                }
            )
            
            # Check if we got a parsed response
            if hasattr(response, "parsed"):
                return response.parsed
            else:
                raise Exception("Failed to parse response")
                
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                backoff_time = initial_backoff * (2 ** (retry_count - 1))
                print(f"Retrying in {backoff_time}s... ({retry_count}/{max_retries})")
                time.sleep(backoff_time)
            else:
                print(f"Failed after {max_retries} retries: {str(e)}")
                return None
    
    return None

# --- Example Usage ---
if __name__ == "__main__":
    example_text = "The quick brown fox jumps over the lazy dog."
    #example_text = "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?"
    # example_text = "Google Search is a web search engine developed by Google LLC."
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging text: \"{example_text}\"")

    tagged_result = tag_sentences_ud(example_text)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for s in tagged_result.sentences:
            for token_obj in s.tokens:
                token = token_obj.token
                tag = token_obj.pos_tag
                # Handle potential None for pos_tag if model couldn't assign one
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
                print("----------------------")
    else:
        print("\nFailed to get POS tagging results.")