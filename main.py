import re
import numpy as np
import sentencepiece
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')

corpus=[
    "Artificial Intelligence is the simulation of human intelligence by machines to perform tasks such as reasoning, learning, and decision-making.",
    "IoT (Internet of Things) is a network of interconnected devices that communicate and exchange data with each other over the internet to automate tasks and improve efficiency..",
    "ADAS performs intelligent tasks by assisting drivers with tasks like braking, steering, and monitoring surroundings.",
    "Natural Language Processing in ADAS and IoT is used for voice recognition and commands, enabling users to interact with systems hands-free for navigation, communication, and vehicle controls.",
    "Reinforcement learning involves training agents to make sequences of decisions to maximize rewards."
]

questions=[
    "What is Artificial Intelligence?",
    "What is IoT?",
    "What does ADAS system do?",
    "In what ways the Natural Language Processing used in both Iot and ADAS?",
    "What is reinforcement learning?"
]

def preprocess_text(text):
    text=re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text=text.lower().strip()
    return text

corpus_cleaned=[preprocess_text(doc) for doc in corpus]
questions_cleaned=[preprocess_text(q) for q in questions]
tfidf_vectorizer=TfidfVectorizer()
corpus_vectors=tfidf_vectorizer.fit_transform(corpus_cleaned)

def retrieve_document(question):
    question_vector=tfidf_vectorizer.transform([question])
    similarities=cosine_similarity(question_vector, corpus_vectors).flatten()
    most_similar_idx=np.argmax(similarities)
    return corpus[most_similar_idx]

model_name="t5-small"
tokenizer=T5Tokenizer.from_pretrained(model_name)
model=T5ForConditionalGeneration.from_pretrained(model_name)

def generate_answer(question, context):
    input_text=f"question: {question} context: {context}"
    input_ids=tokenizer.encode(input_text, return_tensors="pt")
    outputs=model.generate(input_ids, max_length=100, num_beams=5, top_p=0.95, temperature=0.7, do_sample=True, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def exact_match(predicted_answer, expected_answer):
    return 1 if predicted_answer.strip().lower() == expected_answer.strip().lower() else 0

def f1(predicted_answer, expected_answer):
    predicted_tokens=nltk.word_tokenize(predicted_answer.lower())
    expected_tokens=nltk.word_tokenize(expected_answer.lower())
    precision=len(set(predicted_tokens).intersection(set(expected_tokens))) / len(predicted_tokens) if predicted_tokens else 0
    recall =len(set(predicted_tokens).intersection(set(expected_tokens))) / len(expected_tokens) if expected_tokens else 0
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0
    
if __name__ == "__main__":
    example_question="In what ways the Natural Language Processing used in both Iot and ADAS?"
    cleaned_question=preprocess_text(example_question)
    retrieved_doc=retrieve_document(cleaned_question)
    print("Question:", example_question)
    print("Retrieved Document:", retrieved_doc)
    answer=generate_answer(example_question, retrieved_doc)
    print("Generated Answer:", answer)
    em_score=exact_match(answer, retrieved_doc)
    f1_score_value=f1(answer, retrieved_doc)
    print("Exact Match Score:", em_score)
    print("F1 Score:", f1_score_value)
