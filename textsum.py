import nltk
import spacy
import string
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure necessary NLTK downloads are available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    STOPWORDS = set()

# Load models with error handling
try:
    # Load spaCy model for NER and linguistic features
    nlp = spacy.load("en_core_web_sm")
    
    # Load sentence transformer model for semantic similarity
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load T5 model for abstractive summarization
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    models_loaded = True
except Exception as e:
    logger.error(f"Failed to load one or more models: {e}")
    models_loaded = False

def extract_keywords(text, top_n=10):
    """Extract key terms using TF-IDF."""
    # Tokenize and clean text
    doc = nlp(text)
    processed_text = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct and token.text.strip()]
    
    # If text is too short, return all terms
    if len(processed_text) <= top_n:
        return set(processed_text)
    
    # Create a document-term matrix with TF-IDF weights
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)  # Include bigrams for better context
    )
    
    # Split text into paragraphs for better keyword diversity
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [' '.join(sent_tokenize(text)[i:i+3]) for i in range(0, len(sent_tokenize(text)), 3)]
    
    try:
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across all paragraphs to get overall importance
        word_scores = np.sum(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = word_scores.argsort()[-top_n:][::-1]
        keywords = {feature_names[i] for i in top_indices}
        
        # Add named entities as they're often important
        entities = get_named_entities(text)
        keywords.update(entities)
        
        return keywords
    except:
        # Fallback if TF-IDF fails (e.g., with very short text)
        return set([token for token in processed_text if len(token) > 3])

def get_named_entities(text):
    """Extract named entities using spaCy with improved filtering."""
    doc = nlp(text)
    
    # Filter entities to remove very common or likely incorrect entities
    filtered_entities = set()
    for ent in doc.ents:
        # Skip very short entities as they're often errors
        if len(ent.text) <= 2:
            continue
            
        # Skip entities that are just numbers
        if ent.text.strip().replace(',', '').replace('.', '').isdigit():
            continue
            
        # Skip entities that are just stopwords
        if ent.text.lower() in STOPWORDS:
            continue
            
        filtered_entities.add(ent.text)
        
    return filtered_entities

def score_sentences(text):
    """Score sentences based on multiple features for importance ranking."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return [], []
    
    # 1. Position scoring - first and last sentences often important
    position_scores = np.zeros(len(sentences))
    for i in range(len(sentences)):
        # First 3 sentences get high scores
        if i < 3:
            position_scores[i] = 1.0 - (i * 0.2)
        # Last 2 sentences get high scores
        elif i >= len(sentences) - 2:
            position_scores[i] = 0.8
    
    # 2. TF-IDF based content scoring
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        content_scores = [tfidf_matrix[i].sum() for i in range(len(sentences))]
        
        # Normalize content scores
        max_score = max(content_scores) if content_scores else 1
        content_scores = [score/max_score for score in content_scores]
    except:
        # Fallback for very short texts
        content_scores = [1.0] * len(sentences)
    
    # 3. Entity presence scoring
    entity_scores = np.zeros(len(sentences))
    entities = get_named_entities(text)
    for i, sentence in enumerate(sentences):
        # Count entities in each sentence
        sent_entities = get_named_entities(sentence)
        if entities:  # Avoid division by zero
            entity_scores[i] = len(sent_entities) / len(entities)
    
    # 4. Length penalization (very short or very long sentences)
    length_scores = np.zeros(len(sentences))
    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)
    for i, length in enumerate(lengths):
        # Penalize sentences that are too short or too long
        if length < 4:
            length_scores[i] = 0.5  # Short sentence penalty
        elif length > avg_length * 2:
            length_scores[i] = 0.7  # Long sentence penalty
        else:
            length_scores[i] = 1.0
    
    # Combine all scores (weighted)
    final_scores = (0.3 * np.array(position_scores) + 
                   0.4 * np.array(content_scores) + 
                   0.2 * np.array(entity_scores) +
                   0.1 * np.array(length_scores))
    
    # Return both sentences and scores
    return sentences, final_scores

def compress_sentence(sentence):
    """Compress individual sentences by removing less important elements."""
    doc = nlp(sentence)
    
    # Words to potentially remove
    removable = ['actually', 'basically', 'really', 'very', 'quite', 'literally', 'definitely']
    
    # Remove certain modifiers and keep essential parts
    tokens_to_keep = []
    for token in doc:
        # Keep if not a removable adverb or determiner (except 'the' for named entities)
        if ((token.pos_ not in ['ADV', 'DET'] or 
            (token.pos_ == 'DET' and token.text.lower() == 'the' and token.head.ent_type_)) and
            token.text.lower() not in removable):
            tokens_to_keep.append(token.text)
    
    compressed = ' '.join(tokens_to_keep)
    return compressed

def ensure_coherence(sentences, min_sentences=3):
    """Ensure coherence by checking for pronouns without references and adding connection words."""
    if len(sentences) <= 1:
        return sentences
        
    # Cohesive devices to add when needed
    connectors = {
        'addition': ['Additionally', 'Furthermore', 'Moreover', 'Also'],
        'contrast': ['However', 'Nevertheless', 'On the other hand', 'In contrast'],
        'cause': ['Therefore', 'As a result', 'Consequently', 'Thus'],
        'example': ['For instance', 'For example', 'Specifically'],
        'time': ['Meanwhile', 'Subsequently', 'Later', 'Following this']
    }
    
    coherent_sentences = [sentences[0]]  # Start with the first sentence
    
    # Fix pronouns without antecedents and add connectors
    for i in range(1, len(sentences)):
        current = sentences[i]
        
        # Check if sentence starts with a pronoun that might need context
        pronoun_start = re.match(r'^(He|She|It|They|This|That|These|Those|Their|His|Her)\b', current, re.IGNORECASE)
        
        # If it starts with a pronoun and is not continuing a clear topic, modify it
        if pronoun_start and i > 1:
            # Do NER on previous sentence to find potential entities to reference
            prev_doc = nlp(sentences[i-1])
            prev_entities = [ent.text for ent in prev_doc.ents if ent.label_ in ('PERSON', 'ORG', 'GPE')]
            
            if prev_entities:
                # Replace the pronoun with the most relevant entity
                current = re.sub(r'^(He|She|It|They|This|That|These|Those|Their|His|Her)\b', 
                                prev_entities[0], current, flags=re.IGNORECASE)
        
        # Determine if we need a connector based on relation to previous sentence
        prev_doc = nlp(coherent_sentences[-1])
        current_doc = nlp(current)
        
        # Simple heuristic for relationship between sentences
        needs_connector = True
        
        # If current sentence references previous entities, less likely to need connector
        prev_entities = {ent.text.lower() for ent in prev_doc.ents}
        current_entities = {ent.text.lower() for ent in current_doc.ents}
        if prev_entities.intersection(current_entities):
            needs_connector = False
            
        # If there's clear thematic continuity, no connector needed
        prev_lemmas = {token.lemma_.lower() for token in prev_doc if token.is_alpha}
        current_lemmas = {token.lemma_.lower() for token in current_doc if token.is_alpha}
        if len(prev_lemmas.intersection(current_lemmas)) > 2:
            needs_connector = False
        
        # Add connector if needed
        if needs_connector and len(current) > 10 and not re.match(r'^(And|But|Or|So|Because|Although|However|Moreover|Therefore|Thus)\b', current, re.IGNORECASE):
            # Choose appropriate connector type based on content
            connector_type = 'addition'  # Default
            
            # Simple heuristics to determine relationship
            if any(word in current.lower() for word in ['but', 'however', 'although', 'despite', 'though']):
                connector_type = 'contrast'
            elif any(word in current.lower() for word in ['because', 'cause', 'reason']):
                connector_type = 'cause'
            elif any(word in current.lower() for word in ['example', 'instance', 'illustrate']):
                connector_type = 'example'
            
            # Add connector at beginning of sentence
            import random
            connector = random.choice(connectors[connector_type])
            current = f"{connector}, {current[0].lower() + current[1:]}"
        
        coherent_sentences.append(current)
    
    # If we have too few sentences, try to split long ones
    if len(coherent_sentences) < min_sentences:
        expanded = []
        for sent in coherent_sentences:
            if len(sent.split()) > 20:
                # Try to split long sentence in half at a comma or conjunction
                mid_point = len(sent) // 2
                split_point = sent.find(', ', mid_point - 20, mid_point + 20)
                if split_point != -1:
                    expanded.append(sent[:split_point+1])
                    expanded.append(sent[split_point+2:])
                else:
                    expanded.append(sent)
            else:
                expanded.append(sent)
        coherent_sentences = expanded
    
    return coherent_sentences

def check_semantic_coverage(original_text, summary):
    """Ensure the summary covers the key semantic areas of the original text."""
    # Get sentence embeddings for original and summary
    original_sentences = sent_tokenize(original_text)
    summary_sentences = sent_tokenize(summary)
    
    if not original_sentences or not summary_sentences:
        return summary
    
    # Create embeddings
    try:
        original_embeddings = sentence_model.encode(original_sentences)
        summary_embeddings = sentence_model.encode(summary_sentences)
        
        # Calculate coverage by measuring similarity between summary and original
        similarities = cosine_similarity(summary_embeddings, original_embeddings)
        
        # For each original sentence, check if it's represented in summary
        max_similarities = np.max(similarities, axis=0)
        uncovered_indices = np.where(max_similarities < 0.6)[0]
        
        # If some important content isn't covered, add most important missing sentences
        if len(uncovered_indices) > 0:
            # Score the uncovered sentences
            uncovered_sentences = [original_sentences[i] for i in uncovered_indices]
            _, uncovered_scores = score_sentences(' '.join(uncovered_sentences))
            
            # Get the most important uncovered sentence
            if len(uncovered_scores) > 0:
                best_idx = np.argmax(uncovered_scores)
                missing_info = uncovered_sentences[best_idx]
                
                # Add it to summary
                if missing_info not in summary:
                    summary += f" {missing_info}"
    except Exception as e:
        logger.warning(f"Semantic coverage check failed: {e}")
    
    return summary

def summarize_text(text, compression_ratio=0.3):
    """Generate an abstractive summary with improved coherence and coverage."""
    if not models_loaded:
        return "Models failed to load. Cannot generate summary."
    
    if not text or len(text.strip()) < 50:
        return "Input text too short for meaningful summarization."
    
    # Step 1: Extract key information
    try:
        keywords = extract_keywords(text)
        
        sentences, importance_scores = score_sentences(text)
        if not sentences:
            return "Could not parse sentences from input."
            
        # Extract top sentences based on importance for extractive summary
        num_sentences = max(3, int(len(sentences) * compression_ratio))
        top_indices = importance_scores.argsort()[-num_sentences:][::-1]
        
        # Sort by original position to maintain flow
        top_indices.sort()
        extractive_summary = [sentences[i] for i in top_indices]
        
        # Ensure coherence in the selected sentences
        coherent_extractive = ensure_coherence(extractive_summary)
        extractive_text = ' '.join(coherent_extractive)
        
        # Use extractive summary to guide abstractive summarization
        prefix = "summarize: "
        
        # Create a prompt that emphasizes important elements
        keyword_str = ' '.join(keywords)
        input_text = f"{prefix}{extractive_text} Key concepts: {keyword_str}"
        
        # Encode input for T5
        input_ids = t5_tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # Generate abstractive summary
        summary_ids = t5_model.generate(
            input_ids,
            max_length=150,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        
        abstractive_summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Check and improve semantic coverage
        final_summary = check_semantic_coverage(text, abstractive_summary)
        
        # Ensure target compression ratio is met
        target_length = int(len(text.split()) * compression_ratio)
        summary_words = len(final_summary.split())
        
        if summary_words > target_length:
            # Compress each sentence
            summary_sentences = sent_tokenize(final_summary)
            compressed_sentences = [compress_sentence(s) for s in summary_sentences]
            
            # Sort by importance and keep only what fits in target length
            sentence_importances = []
            for i, sentence in enumerate(compressed_sentences):
                # Calculate importance based on keywords and entities
                importance = sum(1 for k in keywords if k.lower() in sentence.lower())
                sentence_importances.append((i, importance, len(sentence.split())))
            
            # Sort by importance-to-length ratio (descending)
            sentence_importances.sort(key=lambda x: x[1]/max(1, x[2]), reverse=True)
            
            # Keep sentences until we hit target length
            final_sentences = []
            current_length = 0
            for i, imp, length in sentence_importances:
                if current_length + length <= target_length:
                    final_sentences.append((i, compressed_sentences[i]))
                    current_length += length
            
            # Sort back to original order
            final_sentences.sort(key=lambda x: x[0])
            final_summary = ' '.join(s for i, s in final_sentences)
        
        # Fix capitalization and spacing issues
        sentences = sent_tokenize(final_summary)
        sentences = [s[0].upper() + s[1:] if len(s) > 0 else s for s in sentences]
        final_summary = ' '.join(sentences)

        # Remove the awkward "Important concepts include" text if present
        if "Important concepts include" in final_summary:
            final_summary = final_summary.split("Important concepts include")[0].strip()
        
        return final_summary
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"An error occurred during summarization: {str(e)}"

# Example usage function
def run_summarization():
    print("Enhanced Text Summarization System")
    print("=================================")
    
    # Get user input for text
    input_text = input("Enter the text to summarize (or type 'file' to load from file):\n")
    
    if input_text.lower().strip() == 'file':
        file_path = input("Enter the path to your text file: ")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            print(f"Loaded {len(input_text)} characters from file.")
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    
    # Validate input
    if not input_text or len(input_text.strip()) < 50:
        print("Input text is too short for meaningful summarization (minimum 50 characters).")
        return
    
    # Get user input for compression ratio
    try:
        user_ratio = float(input("Enter the compression ratio (0.1-0.5, default 0.3): ") or 0.3)
        if not (0.1 <= user_ratio <= 0.5):
            print("Invalid ratio! Using default (0.3).")
            user_ratio = 0.3
    except ValueError:
        print("Invalid ratio! Using default (0.3).")
        user_ratio = 0.3
    
    print("\nAnalyzing text and generating summary...")
    
    # Generate summary
    summary = summarize_text(input_text, compression_ratio=user_ratio)
    
    # Print original length and summary length
    original_words = len(input_text.split())
    summary_words = len(summary.split())
    compression = round((1 - (summary_words / original_words)) * 100)
    
    print("\n" + "="*60)
    print(f"SUMMARY ({summary_words} words, {compression}% compression):")
    print("="*60)
    print(summary)
    print("="*60)

if __name__ == "__main__":
    run_summarization()