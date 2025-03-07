import nltk
import spacy
import string
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ensure necessary NLTK downloads are available
nltk.download('punkt', quiet=True)

# Load spaCy English model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Load T5 model for abstractive summarization
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def get_named_entities(text):
    """Extract named entities using spaCy for better accuracy."""
    doc = nlp(text)
    return {ent.text.lower() for ent in doc.ents}

def summarize_text(text, compression_ratio=0.3):
    """Summarize text using T5 transformer model with optimized settings for better abstraction."""
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return "No sentences found in input."
    
    num_sentences = max(1, int(len(sentences) * compression_ratio))  # Ensure at least 1 sentence
    
    # Named Entity Recognition (NER) to emphasize key elements
    named_entities = get_named_entities(text)
    
    # Modify the input prompt for better abstraction
    t5_input_text = "Provide a concise summary: " + text
    input_ids = t5_tokenizer.encode(t5_input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary with optimized parameters
    summary_ids = t5_model.generate(
        input_ids,
        max_length=150,
        num_beams=5,  # Increase beams for better coherence
        do_sample=True,  # Enable sampling for diversity
        top_k=80,  # Allow more varied words
        temperature=0.9,  # Increase randomness for more abstraction
        repetition_penalty=1.2,  # Reduce repetition
        early_stopping=True
    )
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Ensure important named entities are retained
    for entity in named_entities:
        if entity.lower() not in summary.lower():
            summary += f" ({entity})"
    
    return summary

# Get user input for text
input_text = input("Enter the text to summarize:\n")

# Get user input for compression ratio (optional)
try:
    user_ratio = float(input("Enter the compression ratio (0-1, default 0.3): ") or 0.3)
    if not (0 < user_ratio <= 1):
        raise ValueError
except ValueError:
    print("Invalid ratio! Using default (0.3).")
    user_ratio = 0.3

# Generate summary
summary = summarize_text(input_text, compression_ratio=user_ratio)

# Print the generated summary
print("\nGenerated Summary:")
print(summary)
