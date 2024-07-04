import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Uncomment the following lines if you haven't already downloaded the NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

keywords_to_consider = input("Enter the keywords in your search query: ").split()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)

documents = []
authors = []
ratings = []
publish_dates = []

for filename in os.listdir():
    if filename.endswith(".txt"):
        with open(filename, 'r') as file:
            text = file.read()
            lines = text.split('\n')
            author = None
            rating = None
            publish_date = None

            if len(lines) >= 2:
                author_match = re.match(r'Author:(.*)', lines[0])
                if author_match:
                    author = author_match.group(1).strip()
                rating_match = re.match(r'.*\(([\d.]+)\)', lines[0])
                if rating_match:
                    rating = float(rating_match.group(1))
                date_match = re.match(r'Date of Publishing : (.*)', lines[1])
                if date_match:
                    publish_date = date_match.group(1).strip()

            authors.append(author)
            ratings.append(rating)
            publish_dates.append(publish_date)
            processed_text = preprocess_text("\n".join(lines[2:]))
            documents.append(processed_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

document_lengths = [len(doc.split()) for doc in documents]
max_document_length = max(document_lengths)

normalized_document_lengths = [length / max_document_length for length in document_lengths]

keyword_counts = []
for doc in documents:
    keyword_count = sum(1 for token in word_tokenize(doc) if token in keywords_to_consider)
    keyword_counts.append(keyword_count)

weights = {
    "tfidf": 0.3,
    "doc_length": 0.1,
    "keyword_count": 0.2,
    "author_credibility": 0.2,
    "publish_date": 0.2
}

min_publish_date = min([int(date.replace('-', '')) for date in publish_dates if date is not None])
max_publish_date = max([int(date.replace('-', '')) for date in publish_dates if date is not None])
normalized_publish_dates = [5 + 5 * (int(date.replace('-', '')) - min_publish_date) / (max_publish_date - min_publish_date) if date is not None else None
    for date in publish_dates
]

for factor in ["tfidf", "doc_length", "keyword_count", "author_credibility", "publish_date"]:
    factor_scores = []
    for i in range(len(documents)):
        if factor == "tfidf":
            factor_scores.append(sum(tfidf_matrix[i].toarray()[0]))
        elif factor == "doc_length":
            factor_scores.append(sum(tfidf_matrix[i].toarray()[0]) * normalized_document_lengths[i])
        elif factor == "keyword_count":
            factor_scores.append(keyword_counts[i])
        elif factor == "author_credibility":
            factor_scores.append(ratings[i] if ratings[i] is not None else 0.0)
        elif factor == "publish_date":
            factor_scores.append(normalized_publish_dates[i])
    
    sorted_documents_by_factor = sorted(zip(os.listdir(), factor_scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nRanking based on {factor}:")
    for i, (filename, score) in enumerate(sorted_documents_by_factor):
        print(f"{i+1}. {filename} - Score: {score:.2f}")

aggregated_scores = []
for i in range(len(documents)):
    tfidf_score = sum(tfidf_matrix[i].toarray()[0])
    doc_length_normalized_score = tfidf_score * normalized_document_lengths[i]
    keyword_count_score = keyword_counts[i]
    author_rating = ratings[i] if ratings[i] is not None else 0.0
    publish_date_score = normalized_publish_dates[i]
    aggregated_score = (
        weights["tfidf"] * tfidf_score +
        weights["doc_length"] * doc_length_normalized_score +
        weights["keyword_count"] * keyword_count_score +
        weights["author_credibility"] * author_rating +
        weights["publish_date"] * publish_date_score
    )
    aggregated_scores.append(aggregated_score)

sorted_documents = sorted(zip(os.listdir(), aggregated_scores), key=lambda x: x[1], reverse=True)

print("\nFinal Ranking: ")
for i, (filename, score) in enumerate(sorted_documents):
    print(f"{i+1}. {filename} - Score: {score:.2f}")
