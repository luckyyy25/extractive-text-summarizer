# Enhanced Text Summarization System

An advanced Python-based text summarization tool that combines extractive and abstractive summarization techniques to generate coherent, comprehensive summaries of long texts.

## Features

- **Hybrid Summarization**: Combines extractive and abstractive approaches for optimal results
- **Multi-factor Sentence Scoring**: Uses position, content relevance, entity presence, and length optimization
- **Semantic Coverage**: Ensures important topics from the original text are represented
- **Coherence Enhancement**: Improves flow with connective words and pronoun resolution
- **Keyword Extraction**: Identifies key terms using TF-IDF and named entity recognition
- **Customizable Compression**: Adjustable compression ratios from 10% to 50%
- **Error Handling**: Robust error handling with fallback mechanisms

## Technology Stack

- **NLP Libraries**: NLTK, spaCy
- **Machine Learning**: scikit-learn, sentence-transformers
- **Deep Learning**: Transformers (T5 model), PyTorch
- **Text Processing**: TF-IDF vectorization, cosine similarity

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd text-summarization
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required language models:
```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# NLTK data will be downloaded automatically on first run
```

## Usage

### Interactive Mode

Run the script directly for interactive summarization:

```bash
python textsum.py
```

The program will prompt you to:
- Enter text directly or load from a file
- Specify compression ratio (0.1-0.5, default: 0.3)

### Programmatic Usage

```python
from textsum import summarize_text

# Basic usage
text = "Your long text here..."
summary = summarize_text(text)

# With custom compression ratio
summary = summarize_text(text, compression_ratio=0.2)  # 20% of original length
```

### File Input

When prompted, type `'file'` and provide the path to your text file:
```
Enter the text to summarize (or type 'file' to load from file):
file
Enter the path to your text file: /path/to/your/document.txt
```

## How It Works

### 1. Text Analysis
- **Keyword Extraction**: Uses TF-IDF to identify important terms and phrases
- **Named Entity Recognition**: Extracts people, organizations, and locations
- **Sentence Parsing**: Breaks text into sentences with linguistic analysis

### 2. Sentence Scoring
The system scores sentences based on:
- **Position**: First and last sentences often contain key information
- **Content Relevance**: TF-IDF scores for topical importance
- **Entity Density**: Sentences with more named entities are prioritized
- **Length Optimization**: Penalizes very short or overly long sentences

### 3. Extractive Summarization
- Selects top-scoring sentences based on importance
- Maintains original sentence order for narrative flow
- Ensures minimum sentence count for coherence

### 4. Abstractive Enhancement
- Uses T5 transformer model for text generation
- Incorporates extracted keywords to guide generation
- Applies beam search for quality optimization

### 5. Post-Processing
- **Coherence Enhancement**: Adds connective words and resolves pronouns
- **Semantic Coverage**: Verifies all important topics are covered
- **Compression Control**: Ensures target length is achieved

## Configuration

### Compression Ratios
- **0.1 (10%)**: Very brief summaries, key points only
- **0.2 (20%)**: Concise summaries with main ideas
- **0.3 (30%)**: Balanced summaries (default)
- **0.4 (40%)**: Detailed summaries with context
- **0.5 (50%)**: Comprehensive summaries

### Model Parameters
The system uses several pre-trained models:
- **T5-base**: For abstractive summarization
- **all-MiniLM-L6-v2**: For semantic similarity
- **en_core_web_sm**: For linguistic analysis

## Requirements

See `requirements.txt` for complete dependencies:

```
nltk>=3.8.1
spacy>=3.7.2
transformers>=4.38.2
torch>=2.1.2
scikit-learn>=1.4.1
sentence-transformers>=2.6.1
numpy>=1.26.4
```

## Example Output

```
SUMMARY (45 words, 78% compression):
The research demonstrates significant improvements in natural language processing through transformer architectures. Key findings include enhanced performance on summarization tasks and better semantic understanding. These advances have practical applications in automated content generation and information extraction systems.
```

## Error Handling

The system includes comprehensive error handling:
- Model loading failures with graceful degradation
- Input validation for text length and format
- Fallback mechanisms for processing edge cases
- Detailed logging for debugging

## Performance Considerations

- **Memory Usage**: Models require ~2-4GB RAM
- **Processing Time**: Varies with text length (typically 5-30 seconds)
- **GPU Support**: Automatically uses CUDA if available
- **Batch Processing**: Optimized for single documents

## Limitations

- Minimum input length: 50 characters
- Maximum recommended length: 10,000 words
- English language only (spaCy model dependency)
- Requires internet connection for initial model downloads

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Serkan Kahveci
- Emre Burğucu

## Acknowledgments

- Hugging Face Transformers library
- spaCy team for NLP tools
- NLTK contributors
- Sentence Transformers project
