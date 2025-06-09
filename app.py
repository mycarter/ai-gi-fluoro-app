import os
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import openai
from streamlit_image_viewer import image_viewer

# --- Config ---
st.set_page_config(page_title="GI Fluoro vs CT", layout="wide")
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("matched_gi_ct_fluoro.csv").fillna("")

df = load_data()
img_dir = Path("fluoro_CT images/Gastrointestinal")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters & Search")

# Tag Filter
if "tags" in df.columns:
    tag_list = sorted(set(sum(df["tags"].dropna().apply(lambda x: x.split(", ")), [])))
    tags = st.sidebar.multiselect("Filter by Tag", options=tag_list)
    if tags:
        df = df[df["tags"].apply(lambda t: any(tag in t for tag in tags))]

# Free Text Search
query = st.sidebar.text_input("Search cases")
if query:
    df = df[df.apply(lambda row: query.lower() in str(row).lower(), axis=1)]

# Age Filter
if "age" in df.columns:
    age_range = st.sidebar.slider("Age", 0, 100, (0, 100))
    age_series = pd.to_numeric(df["age"].astype(str).str.extract(r'(\d+)')[0], errors="coerce")
    age_mask = age_series.between(age_range[0], age_range[1], inclusive="both")
    df = df[age_mask.fillna(False)]

# Gender Filter
if "gender" in df.columns:
    genders = st.sidebar.multiselect("Gender", ["Male", "Female"])
    if genders:
        df = df[df["gender"].isin(genders)]

# --- Case Selection ---
st.title("💀 GI Fluoroscopy vs CT Comparison")

if len(df) == 0:
    st.warning("No cases match your filter/search.")
    st.stop()

diagnosis = st.selectbox("Select a diagnosis", df["diagnosis"].unique())
row = df[df["diagnosis"] == diagnosis].iloc[0]

# --- Prepare Images ---
fluoro_key = row["image"].split("_")[0]
fluoro_imgs = sorted([f for f in os.listdir(img_dir) if fluoro_key in f and f.endswith((".jpg", ".png", ".jpeg"))])

ct_key = row["image_ct"].split("_")[0] if row["image_ct"] else None
ct_imgs = sorted([f for f in os.listdir(img_dir) if ct_key and ct_key in f and f.endswith((".jpg", ".png", ".jpeg"))])

# --- Dual View: Fluoro | CT ---
st.subheader("🖼️ Side-by-Side PACS-style Viewer")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔎 Fluoroscopy")
    if fluoro_imgs:
        image_viewer([str(img_dir / f) for f in fluoro_imgs])
        st.markdown(f"**Description:** {row['description']}")
        st.markdown(f"[🩽 View Fluoro Case on Radiopaedia]({row['url']})", unsafe_allow_html=True)

with col2:
    st.markdown("### 🧠 CT")
    if ct_imgs:
        image_viewer([str(img_dir / f) for f in ct_imgs])
        st.markdown(f"**CT Description:** {row['description_ct']}")
        st.markdown(f"[🧠 View CT Case on Radiopaedia]({row['url_ct']})", unsafe_allow_html=True)
    else:
        st.info("No matched CT case available.")

# --- Chatbot ---
mode = st.selectbox("Tutor Mode", ["Teaching", "Quiz Me", "Explain Findings", "Clinical Pearls"])
prompt_map = {
    "Teaching": "You are a tutor. Explain the fluoroscopy and CT findings step-by-step.",
    "Quiz Me": "You are a quiz master. Ask me questions based on this case.",
    "Explain Findings": "List the key radiologic signs in CT and Fluoroscopy for this case.",
    "Clinical Pearls": "Give 3 clinical pearls about this diagnosis and modality use."
}
if st.checkbox("Show Flashcard Example"):
    st.markdown("**Q:** What finding supports SMA syndrome on CT?\n\n**A:** Decreased aortomesenteric angle with duodenal compression.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": prompt_map[mode]}
    ]

st.subheader("💬 AI Tutor Chat")
question = st.text_input("Ask a question about this case...")
if st.button("Ask"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=st.session_state.messages,
            )
            reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.markdown(f"**AI Response:** {reply}")
        except Exception as e:
            st.error(f"API Error: {e}")
