# -*- coding: utf-8 -*-
"""Medical_Specialty_Recommender.py"""

# Install required packages

# Some necessary NLP resources
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Load dataset
df = pd.read_csv("./SpecialtyDescriptions.csv", usecols=["Specialty", "sentence"])
df.dropna(inplace=True)

# Define constants
STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200
PATTERN_S = re.compile("\'s")
PATTERN_RN = re.compile("\\r\\n")
PATTERN_PUNC = re.compile(r"[^\w\s]")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    return text

# Tokenizer function
def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    tokens = [w for w in tokens if (len(w) > min_words and len(w) < max_words and w not in stopwords)]
    return tokens

# Clean and tokenize dataset
def clean_sentences(df):
    df['clean_sentence'] = df['sentence'].apply(clean_text)
    df['tok_lem_sentence'] = df['clean_sentence'].apply(lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True))
    return df

df = clean_sentences(df)

# Utility function for extracting best indices
def extract_best_indices(m, topk, mask=None):
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1]
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask)
    best_index = index[mask][:topk]
    return best_index

# Fit TF-IDF vectorizer
token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=False)
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
tfidf_mat = vectorizer.fit_transform(df['sentence'].values)

# Function to get recommendations
def get_recommendations_tfidf(sentence, tfidf_mat, vectorizer):
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)
    mat = cosine_similarity(vec, tfidf_mat)
    best_index = extract_best_indices(mat, topk=3)
    return best_index

# FastAPI application
app = FastAPI()

# Allow all origins for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    sentence: str

@app.post("/recommend/")
async def recommend_specialty(query: Query):
    try:
        best_index = get_recommendations_tfidf(query.sentence, tfidf_mat, vectorizer)
        recommendations = df['Specialty'].iloc[best_index].tolist()
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

