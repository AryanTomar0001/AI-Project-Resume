# app.py
import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from utils.extractors import extract_text
from models.nlp_models import extract_skills_from_text, tfidf_similarity, load_role_model, generate_recommendations, clean_text
import joblib

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'.pdf', '.docx'}

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Try to load model
role_model, vectorizer = load_role_model()

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_text = request.form.get('job_text', '')
        job_skills_raw = request.form.get('job_skills', '')
        job_skills = [s.strip().lower() for s in job_skills_raw.split(',') if s.strip()]
        file = request.files.get('resume')

        if not file or file.filename == '':
            return render_template('index.html', error='Please upload a resume (.pdf or .docx)')
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return render_template('index.html', error='Unsupported file type. Use .pdf or .docx')

        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Extract resume text
        resume_text = extract_text(path)
        resume_text_clean = clean_text(resume_text)

        # Extract skills
        resume_skills = extract_skills_from_text(resume_text)

        # If job_skills not provided, try to extract from job_text
        if not job_skills and job_text:
            job_skills = extract_skills_from_text(job_text)

        # Similarity score
        if vectorizer:
            sim_score = tfidf_similarity(resume_text, job_text, vectorizer=vectorizer)
        else:
            sim_score = tfidf_similarity(resume_text, job_text)

        sim_percent = round(sim_score * 100, 2)

        # Role prediction
        role_pred = None
        role_conf = None
        if role_model and vectorizer:
            vec = vectorizer.transform([resume_text_clean])
            pred_prob = role_model.predict_proba(vec)[0]
            pred_class = role_model.classes_[pred_prob.argmax()]
            role_pred = pred_class
            role_conf = round(pred_prob.max() * 100, 2)

        missing, suggestions = generate_recommendations(resume_skills, job_skills)

        result = {
            'similarity': sim_percent,
            'extracted_skills': resume_skills,
            'job_skills': job_skills,
            'missing_skills': missing,
            'suggestions': suggestions,
            'resume_filename': filename,
            'role_pred': role_pred,
            'role_conf': role_conf
        }
        return render_template('result.html', result=result)

    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
