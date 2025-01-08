# Meet Pybot: your friendly chatbot

import nltk
import warnings
import numpy as np
import random
import string

# Suppress warnings
warnings.filterwarnings("ignore")

# Load text files
with open(r'C:\Desktop\PyBot\PyBot-A-ChatBot-For-Answering-Python-Queries-Using-NLP-master\nlp_python_answer_finals.txt', 'r', errors='ignore') as f:
    raw = f.read().lower()  # Convert to lowercase

with open(r'C:\Desktop\PyBot\PyBot-A-ChatBot-For-Answering-Python-Queries-Using-NLP-master\modules_pythons.txt', 'r', errors='ignore') as m:
    rawone = m.read().lower()  # Convert to lowercase

# NLTK downloads (uncomment if needed)
# nltk.download('punkt')
# nltk.download('wordnet')

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
sent_tokensone = nltk.sent_tokenize(rawone)
word_tokensone = nltk.word_tokenize(rawone)

# Lemmatization setup
lemmer = nltk.stem.WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = {ord(punct): None for punct in string.punctuation}

def normalize_text(text):
    return lemmatize_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Introductions and greetings
introduce_answers = [
    "My name is PyBot.",
    "You can call me Pi.",
    "I'm PyBot :)",
    "My nickname is Pi, and I'm happy to assist you!"
]

GREETING_INPUTS = ("hello", "hi", "hiii", "hii", "hiiii", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "hello", "I am glad to chat with you"]

basic_questions = {
    "what is python?": "Python is a high-level, interpreted, interactive, and object-oriented programming language designed for readability.",
    "what is module?": [
        "Consider a module to be the same as a code library.",
        "A file containing a set of functions you want to include in your application.",
        "A module can define functions, classes, and variables, making code easier to understand."
    ]
}

# Response functions
def get_greeting(sentence):
    """Return a greeting response if user's input is a greeting."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def get_basic_response(sentence):
    """Return the answer for basic questions."""
    return basic_questions.get(sentence.lower(), None)

def introduce_me():
    """Return an introduction response."""
    return random.choice(introduce_answers)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_response(user_response):
    """Generate a response based on user input."""
    sent_tokens.append(user_response)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=normalize_text, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sent_tokens)

    similarity_values = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)

    idx = similarity_values.argsort()[0][-2]

    if similarity_values.flatten()[idx] == 0:
        return "I am sorry! I don't understand you."

    return sent_tokens[idx]

def chat(user_response):
    """Main chat function to handle user input."""
    user_response = user_response.lower()

    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            return "You are welcome.."
        
        greeting_response = get_greeting(user_response)
        if greeting_response:
            return greeting_response
        
        basic_response = get_basic_response(user_response)
        if basic_response:
            return basic_response
        
        if 'module' in user_response:
            return random.choice(basic_questions["what is module?"])

        return generate_response(user_response)

    return "Bye! Take care.."
