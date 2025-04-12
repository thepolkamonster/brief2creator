import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ------------------
# Caching & Model
# ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-m3")

# ------------------
# Helper Functions
# ------------------
def load_dataset(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Upload a CSV or Excel file.")
    return df

def validate_dataset(df, required_cols):
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset must include the following columns: {required_cols}")

def display_creator_card(row):
    card_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
        <h3 style="margin-bottom: 5px;">{row['name']}</h3>
        <p>
            <strong>Niche:</strong> {row['niche']} |
            <strong>Location:</strong> {row['location']} |
            <strong>Audience:</strong> {row['audience_size']}
        </p>
        <p style="color: #006600;"><strong>Similarity Score:</strong> {row['similarity_score']:.4f}</p>
        <p><em>{row['bio']}</em></p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# -------------------------------
# Streamlit App – Creator Matching
# -------------------------------
st.set_page_config(page_title="Creator Matching App", layout="wide")
st.title("🔍 Creator Matching App")
st.markdown("### Upload your creator dataset (CSV or Excel format)")

uploaded_file = st.file_uploader("📁 Upload your creator dataset", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Please upload your dataset to get started.")
    st.stop()

try:
    creators_df = load_dataset(uploaded_file)
    required_cols = {"name", "bio", "niche", "location", "audience_size"}
    validate_dataset(creators_df, required_cols)
    st.success(f"✅ Dataset loaded successfully with {len(creators_df)} creators.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Store in session
st.session_state.creators_df = creators_df

# Input for campaign brief
st.markdown("### Enter your campaign brief below:")
campaign_brief = st.text_area("Campaign Brief", placeholder="e.g. Looking for Indian influencers who focus on eco-friendly fashion and modern lifestyle.")

if campaign_brief.strip() == "":
    st.info("Please enter a campaign brief to find matches.")
    st.stop()

# Load model once
with st.spinner("Loading embedding model..."):
    model = load_model()

# Embed bios once
if "bio_embedding" not in creators_df.columns:
    with st.spinner("Embedding creator bios..."):
        bio_texts = creators_df["bio"].tolist()
        bio_embeddings = model.encode(bio_texts, convert_to_tensor=False, normalize_embeddings=True)
        creators_df["bio_embedding"] = bio_embeddings.tolist()

# Embed query if changed
if "query_embedding" not in st.session_state or st.session_state.get("last_query", "") != campaign_brief:
    with st.spinner("Embedding campaign brief..."):
        query = "Represent this sentence for searching relevant passages: " + campaign_brief
        query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        st.session_state.query_embedding = query_embedding
        st.session_state.last_query = campaign_brief
else:
    query_embedding = st.session_state.query_embedding

# Compute similarity if not done for this query
if "similarity_score" not in creators_df.columns or st.session_state.last_query != campaign_brief:
    with st.spinner("Calculating similarity scores..."):
        device = query_embedding.device
        creator_embeddings_tensor = torch.tensor(creators_df["bio_embedding"].tolist(), device=device)
        similarities = util.cos_sim(query_embedding, creator_embeddings_tensor)[0].cpu().numpy()
        creators_df["similarity_score"] = similarities

# Filter UI
st.sidebar.header("🔧 Filter Results")
unique_niches = sorted(creators_df["niche"].unique())
unique_locations = sorted(creators_df["location"].unique())

selected_niche = st.sidebar.selectbox("Filter by Niche", ["All"] + unique_niches)
selected_location = st.sidebar.selectbox("Filter by Location", ["All"] + unique_locations)

# Apply filters
filtered_df = creators_df.copy()
if selected_niche != "All":
    filtered_df = filtered_df[filtered_df["niche"] == selected_niche]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df["location"] == selected_location]

# Sort and display
top_creators = filtered_df.sort_values(by="similarity_score", ascending=False).head(10)

st.markdown("### 🎯 Top Matching Creators")
if top_creators.empty:
    st.warning("No matching creators found. Adjust filters or try a different brief.")
else:
    for _, row in top_creators.iterrows():
        display_creator_card(row)

# Download option
st.markdown("### 📥 Download Results")
csv_data = top_creators.to_csv(index=False)
st.download_button(label="Download CSV", data=csv_data, file_name="top_creators.csv", mime="text/csv")
