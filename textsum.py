import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

def get_named_entities(text):
    """Extract named entities from the text to prioritize key sentences."""
    words = word_tokenize(text)
    tagged = pos_tag(words)
    chunked = ne_chunk(tagged)
    named_entities = {word for subtree in chunked if hasattr(subtree, 'label') for word, _ in subtree.leaves()}
    return named_entities

def summarize_text(text, compression_ratio=0.3):
    """Summarize text using TF-IDF weighting and Named Entity Recognition (NER) boosting."""
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return "No sentences found in input."

    num_sentences = max(1, int(len(sentences) * compression_ratio))  # Ensure at least 1 sentence

    # Stopwords & Punctuation Removal
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    # Named Entity Recognition (NER)
    named_entities = get_named_entities(text)

    # Compute TF-IDF scores for words
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()  # Sum scores per sentence

    # Score sentences based on TF-IDF and Named Entity presence
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        sentence_scores[i] = tfidf_scores[i]  # Base score from TF-IDF
        
        # Boost score if sentence contains named entities
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