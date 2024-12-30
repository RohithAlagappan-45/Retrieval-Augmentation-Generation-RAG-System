import os
import re
import nltk
import numpy as np
import sentencepiece
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from docx import Document
nltk.download('punkt')

def preprocess_text(text):
    text=re.sub(r'[^a-zA-Z0-9 ]','',text)
    text=text.lower().strip()
    return text

def extract_text_from_word(file_path):
    doc=Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_documents(folder_path):
    corpus=[]
    for file_name in os.listdir(folder_path):
        file_path=os.path.join(folder_path, file_name)
        if file_name.endswith('.docx'):
            corpus.append(extract_text_from_word(file_path))
    return corpus

folder_path="D:/Fasttack/files" 
raw_corpus=load_documents(folder_path)
corpus_cleaned=[preprocess_text(doc) for doc in raw_corpus]

questions=[
    "What is Artificial Intelligence?",
    "How does IoT work?",
    "Explain reinforcement learning.",
    "What is the purpose of ADAS?",
    "How is NLP used in modern systems?"
]

questions_cleaned=[preprocess_text(q) for q in questions]

tfidf_vectorizer=TfidfVectorizer()
corpus_vectors=tfidf_vectorizer.fit_transform(corpus_cleaned)

def retrieve_document(question):
    question_vector=tfidf_vectorizer.transform([question])
    similarities=cosine_similarity(question_vector, corpus_vectors).flatten()
    most_similar_idx=np.argmax(similarities)
    return raw_corpus[most_similar_idx]

model_name="t5-small"
tokenizer=T5Tokenizer.from_pretrained(model_name)
model=T5ForConditionalGeneration.from_pretrained(model_name)

def generate_answer(question, context):
    input_text=f"question: {question} context: {context}"
    input_ids=tokenizer.encode(input_text, return_tensors="pt")
    outputs=model.generate(input_ids, max_length=50, num_beams=5, top_p=0.95, temperature=0.7, do_sample=True, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def exact_match(predicted_answer, expected_answer):
    return 1 if predicted_answer.strip().lower() == expected_answer.strip().lower() else 0

def f1(predicted_answer, expected_answer):
    predicted_tokens=nltk.word_tokenize(predicted_answer.lower())
    expected_tokens=nltk.word_tokenize(expected_answer.lower())
    precision=len(set(predicted_tokens).intersection(set(expected_tokens))) / len(predicted_tokens) if predicted_tokens else 0
    recall=len(set(predicted_tokens).intersection(set(expected_tokens))) / len(expected_tokens) if expected_tokens else 0
    if precision+ recall>0:
        return 2*(precision*recall)/(precision+recall)
    else:
        return 0
    
if __name__=="__main__":
    for question in questions:
        cleaned_question=preprocess_text(question)
        retrieved_doc=retrieve_document(cleaned_question)

        relevant_context=""
        for para in retrieved_doc.split("\n"):
            if any(word in para.lower() for word in cleaned_question.split()):
                relevant_context += para + " "
        
        print("Question:",question)
        print("Relevant Context:",relevant_context.strip())
        answer=generate_answer(question,relevant_context.strip())
        print("Generated Answer:",answer)
        em_score=exact_match(answer,retrieved_doc)
        f1_score_value=f1(answer,retrieved_doc)
        print("Exact Match Score:",em_score)
        print("F1 Score:",f1_score_value)
