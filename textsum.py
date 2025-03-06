import nltk
import spacy
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK downloads are available
nltk_dependencies = ['punkt', 'stopwords', 'wordnet']
for dep in nltk_dependencies:
    nltk.download(dep, quiet=True)

# Load spaCy English model for advanced Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

def get_named_entities(text):
    """Extract named entities using spaCy for better accuracy."""
    doc = nlp(text)
    return {ent.text.lower() for ent in doc.ents}

def preprocess_text(text):
    """Tokenize, remove stopwords, punctuation, and apply lemmatization."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(words)

def summarize_text(text, compression_ratio=0.3):
    """Summarize text using TF-IDF and Named Entity Recognition (NER)."""
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return "No sentences found in input."
    
    num_sentences = max(1, int(len(sentences) * compression_ratio))  # Ensure at least 1 sentence
    named_entities = get_named_entities(text)
    
    # Preprocess sentences
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()  # Sum scores per sentence
    
    # Score sentences with NER weighting
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words = set(word_tokenize(sentence.lower()))
        sentence_scores[i] = tfidf_scores[i]  # Base score from TF-IDF
        
        # Boost score if the sentence contains named entities
        if any(word in named_entities for word in words):
            sentence_scores[i] *= 1.5  # Increase weight
    
    # Select top-ranked sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences_indices = sorted([index for index, _ in sorted_sentences[:num_sentences]])
    
    # Construct summary
    summary = ' '.join(sentences[i] for i in top_sentences_indices)
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
