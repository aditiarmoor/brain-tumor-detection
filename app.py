import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import io
import plotly.graph_objects as go
import base64
import requests
from datetime import datetime
import tempfile
import sqlite3
import hashlib
import os
# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Brain Tumor Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Global CSS — text visibility + layout
# ---------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-dark:      #0d1117;
    --bg-card:      rgba(22, 30, 46, 0.92);
    --bg-card-alt:  rgba(15, 23, 42, 0.85);
    --accent:       #3b82f6;
    --accent-red:   #ef4444;
    --accent-green: #22c55e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border:       rgba(255,255,255,0.08);
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f172a 50%, #1a1f35 100%);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary) !important;
}

/* ── Force ALL text to be visible ── */
p, span, div, label, li, td, th,
.stMarkdown, .stText,
[data-testid="stMarkdownContainer"] * {
    color: var(--text-primary) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10, 15, 28, 0.97) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Sidebar buttons ── */
section[data-testid="stSidebar"] button {
    background: rgba(59, 130, 246, 0.15) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(59,130,246,0.4) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(59, 130, 246, 0.3) !important;
}

/* ── Main buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.5);
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(239,68,68,0.4);
}
.stDownloadButton > button:hover {
    box-shadow: 0 6px 22px rgba(239,68,68,0.6);
    transform: translateY(-2px);
}

/* ── Input fields ── */
.stTextInput > div > div > input {
    background: rgba(30, 41, 59, 0.8) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}
.stTextInput label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(30, 41, 59, 0.8) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.stSelectbox svg { color: var(--text-primary) !important; fill: var(--text-primary) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 2px dashed rgba(59,130,246,0.4) !important;
    border-radius: 12px !important;
    padding: 1rem;
}
[data-testid="stFileUploader"] * { color: var(--text-primary) !important; }
[data-testid="stFileUploader"] button {
    background: rgba(59,130,246,0.2) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(59,130,246,0.5) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}
[data-testid="stDataFrame"] * { color: var(--text-primary) !important; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #ef4444) !important;
    border-radius: 4px;
}
.stProgress > div {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 4px;
}

/* ── Alert boxes (success / error / warning / info) ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}
[data-testid="stAlert"] * { color: var(--text-primary) !important; }

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    background: rgba(22,30,46,0.6) !important;
    border-radius: 12px;
    padding: 0.5rem;
    border: 1px solid var(--border);
}

/* ── Card component ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.card * { color: var(--text-primary) !important; }

/* ── Section header inside card ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Radio buttons ── */
.stRadio label { color: var(--text-primary) !important; }
.stRadio div { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Database Setup
# ---------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()


try:
    c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    conn.commit()
except:
    pass

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT DEFAULT 'user'
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_by TEXT,
    patient_name TEXT,
    patient_id TEXT,
    file_name TEXT,
    prediction TEXT,
    probability REAL,
    risk_level TEXT,
    date TEXT
)
""")

# Migration: add missing columns if predictions table already existed without them
for col, col_type in [
    ("uploaded_by", "TEXT"),
    ("patient_name", "TEXT"),
    ("patient_id",   "TEXT"),
    ("file_name",    "TEXT"),
    ("prediction",   "TEXT"),
    ("probability",  "REAL"),
    ("risk_level",   "TEXT"),
    ("date",         "TEXT"),
]:
    try:
        c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
        conn.commit()
    except:
        pass  # Column already exists
conn.commit()

def save_prediction(uploaded_by, patient_name, patient_id, file_name,
                    prediction, probability, risk_level, date):
    c.execute("""
        INSERT INTO predictions
        (uploaded_by, patient_name, patient_id, file_name, prediction, probability, risk_level, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (uploaded_by, patient_name, patient_id, file_name,
          prediction, probability, risk_level, date))
    conn.commit()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hash(password, hashed_text):
    return make_hash(password) == hashed_text

def add_user(username, password, role="user"):
    c.execute(
        "INSERT INTO users(username, password, role) VALUES (?, ?, ?)",
        (username, make_hash(password), role)
    )
    conn.commit()

def login_user(username, password):
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, make_hash(password))
    )
    return c.fetchone()

# Ensure admin exists
c.execute("SELECT * FROM users WHERE username=?", ("admin",))
admin = c.fetchone()
if not admin:
    add_user("admin", "admin123", "admin")

# ---------------------------
# Session State Init
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = "user"

# ---------------------------
# Auth Screen
# ---------------------------
if not st.session_state.logged_in:

    # Centered auth card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <span style="font-size:3rem;">🧠</span>
            <h1 style="font-size:1.8rem; margin:0.5rem 0 0.2rem; color:#f1f5f9 !important;">
                Brain Tumor Detection
            </h1>
            <p style="color:#94a3b8 !important; font-size:0.95rem;">
                AI-Powered Medical Diagnostic System
            </p>
        </div>
        """, unsafe_allow_html=True)

        menu = st.selectbox("", ["Login", "Register"], label_visibility="collapsed")

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if menu == "Login":
            st.markdown('<p class="section-title">🔐 Sign In</p>', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            if st.button("Login", use_container_width=True):
                result = login_user(username, password)
                if result:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = result[3]
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        elif menu == "Register":
            st.markdown('<p class="section-title">📝 Create Account</p>', unsafe_allow_html=True)
            new_user = st.text_input("Username", placeholder="Choose a username")
            new_password = st.text_input("Password", type="password", placeholder="Choose a password")

            if st.button("Register", use_container_width=True):
                try:
                    add_user(new_user, new_password)
                    st.success("✅ Account created! Please login.")
                except sqlite3.IntegrityError:
                    st.error("⚠️ Username already exists.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ---------------------------
# AFTER LOGIN
# ---------------------------
username = st.session_state.get("username", "User")
role = st.session_state.get("role", "user")

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1rem 0 1.5rem;">
        <div style="font-size:2rem; text-align:center;">🧠</div>
        <div style="text-align:center; margin-top:0.5rem;">
            <span style="font-weight:700; font-size:1rem; color:#f1f5f9;">Brain Tumor AI</span><br>
            <span style="font-size:0.8rem; color:#94a3b8;">Diagnostic System</span>
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08); margin-bottom:1rem;">
    <div style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.3);
                border-radius:10px; padding:0.75rem 1rem; margin-bottom:1rem;">
        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;">Logged in as</div>
        <div style="font-weight:700; color:#f1f5f9; margin-top:0.2rem;">👤 {username}</div>
        <div style="font-size:0.78rem; color:#3b82f6; margin-top:0.15rem; text-transform:capitalize;">
            {'🔐 Administrator' if role == 'admin' else '👥 User'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

    if role == "admin":
        st.markdown("---")
        st.markdown('<div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">Admin Panel</div>', unsafe_allow_html=True)
        admin_action = st.radio("", ["None", "View Users", "View Predictions"], label_visibility="collapsed")

# ---------------------------
# Page Header
# ---------------------------
st.markdown("""
<div class="card" style="border-left: 4px solid #3b82f6;">
    <div style="display:flex; align-items:center; gap:1rem;">
        <span style="font-size:2.5rem;">🧠</span>
        <div>
            <h1 style="margin:0; font-size:1.8rem; color:#f1f5f9 !important;">
                Brain Tumor Detection Dashboard
            </h1>
            <p style="margin:0.25rem 0 0; color:#94a3b8 !important; font-size:0.95rem;">
                Upload MRI images to detect brain tumors and generate AI medical reports
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Admin Panel Actions
# ---------------------------
if role == "admin":
    if admin_action == "View Users":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">👥 All Users</p>', unsafe_allow_html=True)
        df = pd.read_sql_query("SELECT id, username, role FROM users", conn)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif admin_action == "View Predictions":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 All Predictions</p>', unsafe_allow_html=True)
        try:
            df = pd.read_sql_query("""
                SELECT
                    id            AS "ID",
                    uploaded_by   AS "Uploaded By",
                    patient_name  AS "Patient Name",
                    patient_id    AS "Patient ID",
                    file_name     AS "File",
                    prediction    AS "Prediction",
                    probability   AS "Probability (%)",
                    risk_level    AS "Risk Level",
                    date          AS "Date"
                FROM predictions
                ORDER BY id DESC
            """, conn)

            if df.empty:
                st.info("No predictions have been saved yet.")
            else:
                # Summary metrics
                total = len(df)
                tumor_cnt = (df["Prediction"] == "Tumor").sum()
                no_tumor_cnt = total - tumor_cnt

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Scans", total)
                m2.metric("Tumor Detected", tumor_cnt)
                m3.metric("No Tumor", no_tumor_cnt)

                st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)

                # Export
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button(
                    "📥 Export All Predictions (CSV)",
                    csv_buf.getvalue(),
                    "all_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

                # Delete all (admin only)
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear All Predictions", use_container_width=True):
                    c.execute("DELETE FROM predictions")
                    conn.commit()
                    st.success("All predictions cleared.")
                    st.rerun()

        except Exception as e:
            st.error(f"Error loading predictions: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# Determine if admin is on a panel view — if so, skip the scan UI entirely
_admin_viewing = role == "admin" and admin_action in ("View Users", "View Predictions")

# ---------------------------
# PDF Libraries
# ---------------------------
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_brain_tumor_model():
    model = load_model("brain_tumor_model.h5")
    return model

model = load_brain_tumor_model()

threshold = 0.5

# ---------------------------
# Patient Info + Upload + Predictions
# Hidden when admin is viewing Users or Predictions panel
# ---------------------------
if not _admin_viewing:

    if "history" not in st.session_state:
        st.session_state.history = []

    results = []

    # Patient Info card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📋 Patient Information</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        patient_name = st.text_input("Patient Name", placeholder="Full name")
    with col_b:
        patient_id = st.text_input("Patient ID", placeholder="e.g. PT-2024-001")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "📤 Upload MRI Images (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Predictions
    if uploaded_files:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🔬 Individual Predictions</p>', unsafe_allow_html=True)

        for uploaded_file in uploaded_files:

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border-radius:10px;
                        padding:0.75rem 1rem; margin-bottom:0.75rem; border:1px solid rgba(255,255,255,0.06);">
                <span style="font-weight:600; color:#94a3b8; font-size:0.85rem;">FILE:</span>
                <span style="font-weight:700; color:#f1f5f9;"> {uploaded_file.name}</span>
            </div>
            """, unsafe_allow_html=True)

            img = Image.open(uploaded_file)
            col_img, col_result = st.columns([1, 1])

            with col_img:
                st.image(img, caption=f"MRI: {uploaded_file.name}", use_container_width=True)

            img_resized = img.resize((224, 224))
            img_rgb = img_resized.convert("RGB") if img_resized.mode != "RGB" else img_resized
            img_array = np.expand_dims(np.array(img_rgb).astype(np.float32) / 255.0, axis=0)

            prediction = model.predict(img_array)
            probability = prediction[0][0] * 100
            label = "Tumor" if prediction[0][0] > threshold else "No Tumor"

            with col_result:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

                if label == "Tumor":
                    st.error(f"⚠️ **Tumor Detected** — {probability:.2f}% confidence")
                else:
                    st.success(f"✅ **No Tumor Detected** — {probability:.2f}% confidence")

                if probability >= 90:
                    risk_level = "High"
                    st.error("🔴 Risk Level: **HIGH**")
                elif probability >= 70:
                    risk_level = "Moderate"
                    st.warning("🟠 Risk Level: **MODERATE**")
                elif probability >= 50:
                    risk_level = "Low"
                    st.info("🟡 Risk Level: **LOW**")
                else:
                    risk_level = "Minimal"
                    st.success("🟢 Risk Level: **MINIMAL**")

                st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:4px;'>Tumor Probability</p>", unsafe_allow_html=True)
                st.progress(min(int(probability), 100))

                fig = go.Figure(data=[go.Pie(
                    labels=["No Tumor", "Tumor"],
                    values=[100 - probability, probability],
                    hoverinfo="label+percent",
                    textinfo="label+percent",
                    marker=dict(colors=["#3b82f6", "#ef4444"]),
                    hole=0.35
                )])
                fig.update_layout(
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=220,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"),
                    legend=dict(font=dict(color="#f1f5f9"))
                )
                st.plotly_chart(fig, use_container_width=True)

            result = {
                "Patient Name": patient_name,
                "Patient ID": patient_id,
                "File": uploaded_file.name,
                "Prediction": label,
                "Probability (%)": round(probability, 2),
                "Risk Level": risk_level,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result)
            st.session_state.history.append(result)

            save_prediction(
                uploaded_by=st.session_state.username,
                patient_name=patient_name,
                patient_id=patient_id,
                file_name=uploaded_file.name,
                prediction=label,
                probability=round(probability, 2),
                risk_level=risk_level,
                date=result["Date"]
            )
            st.markdown("<hr style='border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Summary Table
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Prediction Summary</p>', unsafe_allow_html=True)
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Overall Distribution
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📈 Overall Prediction Distribution</p>', unsafe_allow_html=True)
        tumor_count = df_results[df_results["Prediction"] == "Tumor"].shape[0]
        no_tumor_count = df_results[df_results["Prediction"] == "No Tumor"].shape[0]
        fig2 = go.Figure(data=[go.Pie(
            labels=["No Tumor", "Tumor"],
            values=[no_tumor_count, tumor_count],
            hoverinfo="label+percent",
            textinfo="label+percent",
            marker=dict(colors=["#3b82f6", "#ef4444"]),
            hole=0.4
        )])
        fig2.update_layout(
            margin=dict(t=10, b=10, l=0, r=0),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            legend=dict(font=dict(color="#f1f5f9"))
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # CSV Download
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">💾 Export Data</p>', unsafe_allow_html=True)
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download CSV Report",
            csv_buffer.getvalue(),
            "brain_tumor_predictions.csv",
            "text/csv",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # PDF Reports
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📄 Medical PDF Reports</p>', unsafe_allow_html=True)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], alignment=TA_CENTER)
        signature_style = ParagraphStyle('SignatureStyle', parent=styles['Normal'], alignment=TA_RIGHT)

        for uploaded_file, result in zip(uploaded_files, results):

            probability_value = result["Probability (%)"]

            if probability_value >= 90:
                recommendation = "<b>AI Recommendation:</b><br/><br/>High confidence detection. Consultation with a neurologist or radiologist is strongly recommended."
            elif probability_value >= 70:
                recommendation = "<b>AI Recommendation:</b><br/><br/>Moderate confidence detection. Professional medical review of the MRI scan is recommended."
            elif probability_value >= 50:
                recommendation = "<b>AI Recommendation:</b><br/><br/>Borderline detection. Further medical evaluation may be helpful."
            else:
                recommendation = "<b>AI Recommendation:</b><br/><br/>Low confidence detection. No strong indication was identified. If symptoms exist, consult a healthcare professional."

            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            elements = []

            elements.append(Paragraph("<font size=20><b>Brain Tumor Analysis Report</b></font>", title_style))
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<font size=12>AI Medical Diagnostic Center</font>", styles['Normal']))
            elements.append(Spacer(1, 20))

            img = Image.open(uploaded_file)
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(temp_img.name)
            elements.append(RLImage(temp_img.name, width=220, height=220))
            elements.append(Spacer(1, 20))

            data = [
                ["Patient Name", result["Patient Name"]],
                ["Patient ID", result["Patient ID"]],
                ["Generated By", st.session_state.username],
                ["File Name", result["File"]],
                ["Prediction", result["Prediction"]],
                ["Model Confidence", f"{result['Probability (%)']}%"],
                ["Risk Level", result["Risk Level"]],
                ["Date", result["Date"]]
            ]
            table = Table(data, colWidths=[150, 300])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica')
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))

            drawing = Drawing(400, 200)
            pie = Pie()
            pie.x, pie.y, pie.width, pie.height = 150, 20, 150, 150
            pie.data = [100 - probability_value, probability_value]
            pie.labels = ["No Tumor", "Tumor"]
            drawing.add(pie)
            elements.append(drawing)
            elements.append(Spacer(1, 20))

            elements.append(Paragraph(recommendation, styles['BodyText']))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(
                "___________________________<br/><b>Dr. AI Diagnostic System</b><br/>Senior Radiology Assistant",
                signature_style
            ))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(
                "<font size=9 color='red'>Disclaimer: This report is AI-generated and should not replace professional medical advice. "
                "Please consult a healthcare professional.</font>",
                styles['Italic']
            ))

            doc.build(elements)
            pdf_buffer.seek(0)

            st.success(f"✅ PDF Ready: {uploaded_file.name}")
            st.download_button(
                label=f"📥 Download Report — {uploaded_file.name}",
                data=pdf_buffer,
                file_name=f"Brain_Tumor_Analysis_{result['File']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # Session History
    if st.session_state.history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🕓 Session History</p>', unsafe_allow_html=True)
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Footer (always visible)
# ---------------------------
st.markdown("""
<div class="card" style="text-align:center; border-top: 1px solid rgba(59,130,246,0.3);">
    <p style="font-size:1.1rem; font-weight:700; color:#f1f5f9 !important; margin:0;">
        🧠 AI-Powered Brain Tumor Detection System
    </p>
    <p style="color:#94a3b8 !important; font-size:0.85rem; margin:0.4rem 0 0;">
        Built with Streamlit · TensorFlow · Plotly · ReportLab · Python
    </p>
</div>
""", unsafe_allow_html=True)
