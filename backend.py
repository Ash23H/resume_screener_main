from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()

def calculate_relevance(job_desc, resume_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_desc, resume_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)

@app.route("/upload", methods=["POST"])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    job_desc = request.form.get("job_description", "")
    
    if file.filename == '' or not job_desc:
        return jsonify({"error": "Missing file or job description"}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    resume_text = extract_text_from_pdf(file_path)
    relevance_score = calculate_relevance(job_desc, resume_text)
    
    return jsonify({
        "resume_text": resume_text[:500],  # Limit preview text
        "relevance_score": relevance_score
    })

if __name__ == "__main__":
    app.run(debug=True)
