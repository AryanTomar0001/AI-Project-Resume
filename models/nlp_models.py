# models/nlp_models.py
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

nlp = spacy.load("en_core_web_sm")

# ---------- Text cleaning ----------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.-]', ' ', text)  # keep + # . - for versions e.g. C++, C#
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# ---------- Skill extraction (improved list + fuzzy style check) ----------
COMMON_SKILLS = [
    'python','java','javascript','html','css','react','node','flask','django',
    'angular','vue','sql','mysql','mongodb','postgresql','git','github',
    'docker','kubernetes','api','rest','json','aws','azure','gcp','linux',
    'pandas','numpy','scikit-learn','tensorflow','keras','pytorch','ml','nlp',
    'data analysis','data science','ai','deep learning','computer vision',
    'communication','problem solving','testing','selenium','junit','jira'
]

def extract_skills_from_text(text: str):
    text_clean = clean_text(text)
    found = []
    for skill in COMMON_SKILLS:
        if skill in text_clean:
            found.append(skill)
    # also try token matching for words like "machine learning" -> 'machine learning'
    return sorted(list(set(found)))

# ---------- Similarity: TF-IDF baseline ----------
def tfidf_similarity(resume_text, job_text, vectorizer=None):
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)
    if not vectorizer:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        vectors = vectorizer.fit_transform([resume_text, job_text])
    else:
        vectors = vectorizer.transform([resume_text, job_text])
    try:
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except Exception:
        score = 0.0
    return float(score)

# ---------- Load saved model & vectorizer ----------
def load_role_model(model_path='models/role_classifier.pkl', vec_path='models/vectorizer.pkl'):
    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    return None, None

# ---------- Generate suggestions ----------
def generate_recommendations(extracted_skills, job_skills):
    extracted = set([s.lower() for s in extracted_skills])
    jobset = set([s.lower() for s in job_skills])
    missing = sorted(list(jobset - extracted))
    suggestions = []
    if missing:
        suggestions.append(f"Add or highlight these skills: {', '.join(missing)}")
    else:
        suggestions.append("Good! Your resume already contains the required skills.")
    return missing, suggestions
