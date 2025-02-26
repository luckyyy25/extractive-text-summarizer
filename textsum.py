import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')  # Gerekli olabilir
nltk.download('wordnet')  # Gerekli olabilir
nltk.download('omw-1.4')  # Gerekli olabilir
nltk.download('punkt_tab')  # Hata mesajına göre eksik olan bu

def summarize_text(text, num_sentences=1):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)


    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]


    word_freq = FreqDist(words)

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word, freq in word_freq.items():
            if word in sentence.lower():
                if i in sentence_scores:
                    sentence_scores[i] += freq
                else:
                    sentence_scores[i] = freq


    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences_indices = [sentence[0] for sentence in sorted_sentences[:num_sentences]]


    top_sentences_indices.sort()

    summary_sentences = [sentences[index] for index in top_sentences_indices]


    summary = ' '.join(summary_sentences)
    return summary


input_text = """
Python programming language is known for its simplicity and readability. It is widely used in web development, data science, artificial intelligence, and more. Python has a large and active community of developers, which makes it easy to find support and resources.

One of the key features of Python is its versatility. It supports both procedural and object-oriented programming paradigms. Python's syntax allows developers to express concepts in fewer lines of code compared to languages like C++ or Java.

The Python community has developed a vast number of libraries and frameworks that simplify various tasks. For example, Django and Flask are popular frameworks for web development, while NumPy and Pandas are widely used for data manipulation and analysis.

In recent years, Python has gained significant traction in the field of artificial intelligence and machine learning. Libraries such as TensorFlow and PyTorch have become go-to tools for building and training machine learning models.

Overall, Python's ease of use, extensive libraries, and community support make it an excellent choice for both beginners and experienced developers.

"""

summary = summarize_text(input_text)
print("Input Text:")
print(input_text)
print("\nGenerated Summary:")
print(summary)
