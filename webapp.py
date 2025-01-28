import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.title("Enhanced Candidate Selection Tool")

st.subheader("NLP-Based Resume Screening with Feedback")
st.caption("Evaluate resumes using advanced techniques and provide actionable feedback.")

# Upload job description and resume
uploadedJD = st.file_uploader("Upload Job Description", type="pdf")
uploadedResume = st.file_uploader("Upload Resume", type="pdf")
click = st.button("Process")

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages])

# TF-IDF Keyword Extraction
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Weighted Cosine Similarity
def weighted_cosine_similarity(JD_txt, resume_txt, weight_skills=0.7, weight_experience=0.3):
    cv = CountVectorizer()
    JD_matrix = cv.fit_transform([JD_txt, resume_txt])
    similarity_score = cosine_similarity(JD_matrix)[0][1]
    final_score = (similarity_score * weight_skills) + (similarity_score * weight_experience)
    return final_score * 100

# Highlight Matched and Missing Words
def highlight_text(JD_txt, resume_txt):
    JD_words = set(JD_txt.split())
    resume_words = set(resume_txt.split())
    matched_words = JD_words.intersection(resume_words)
    missing_words = JD_words - resume_words

    highlighted_resume = resume_txt
    for word in matched_words:
        highlighted_resume = highlighted_resume.replace(word, f"**{word}**")
    for word in missing_words:
        highlighted_resume = highlighted_resume.replace(word, f"<span style='color:red;'>{word}</span>")
    return highlighted_resume

# Feedback Generation
def generate_feedback(JD_txt, resume_txt):
    JD_keywords = extract_keywords(JD_txt)
    resume_keywords = extract_keywords(resume_txt)

    missing_keywords = [word for word in JD_keywords if word not in resume_keywords]
    feedback = []

    if missing_keywords:
        feedback.append(f"Your resume is missing the following important keywords: {', '.join(missing_keywords)}.")
    else:
        feedback.append("Your resume contains all the essential keywords from the job description.")

    experience_required = extract_experience(JD_txt)
    candidate_experience = extract_experience(resume_txt)
    if candidate_experience < experience_required:
        feedback.append(f"Your experience level ({candidate_experience} years) is below the required level ({experience_required} years). Consider emphasizing relevant projects or skills.")
    else:
        feedback.append("Your experience level matches the requirements!")

    return feedback

# Extract Experience from Text
def extract_experience(text):
    experience_pattern = re.compile(r'(\d+)[\s+]years')
    experience_match = experience_pattern.search(text)
    if experience_match:
        return int(experience_match.group(1))
    return 0

# Main logic
if click:
    try:
        job_description = extract_text_from_pdf(uploadedJD)
        resume = extract_text_from_pdf(uploadedResume)

        # Extract and display keywords
        JD_keywords = extract_keywords(job_description)
        resume_keywords = extract_keywords(resume)
        st.write("Job Description Keywords:", JD_keywords)
        st.write("Resume Keywords:", resume_keywords)

        # Weighted Cosine Similarity
        similarity_score = weighted_cosine_similarity(job_description, resume)
        st.write(f"Match Percentage: {round(similarity_score, 2)}%")

        # Highlight matched and missing words
        highlighted_resume = highlight_text(job_description, resume)
        st.markdown("**Highlighted Resume:**", unsafe_allow_html=True)
        st.markdown(highlighted_resume, unsafe_allow_html=True)

        # Feedback for improvement
        feedback = generate_feedback(job_description, resume)
        st.write("**Feedback:**")
        for fb in feedback:
            st.write("- ", fb)

        # Extract and compare experience
        required_experience = extract_experience(job_description)
        candidate_experience = extract_experience(resume)
        st.write(f"Required Experience: {required_experience} years")
        st.write(f"Candidate Experience: {candidate_experience} years")

        if candidate_experience < required_experience:
            st.write("Candidate does not meet the required experience level.")
        else:
            st.write("Candidate meets the required experience level.")

    except Exception as e:
        st.error(f"Error processing files: {e}")

st.caption("~ Enhanced by Chidananda")
