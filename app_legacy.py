import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ------------------
# Helper Functions
# ------------------
def load_dataset(uploaded_file):
    """
    Load a CSV or Excel file from the uploaded file.
    Raises ValueError if the file format is unsupported.
    """
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

def validate_dataset(df, required_cols):
    """
    Check that the loaded dataset has all required columns.
    """
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset must include the following columns: {required_cols}")

def display_creator_card(row):
    """
    Display one creator as a styled HTML card.
    """
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

# File uploader: Only show file uploader until file is provided.
uploaded_file = st.file_uploader("📁 Upload your creator dataset", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Please upload your dataset to get started.")
    st.stop()

# Try to load and validate the dataset.
try:
    creators_df = load_dataset(uploaded_file)
    st.success(f"✅ Dataset loaded successfully with {len(creators_df)} creators.")
    
    # Required columns for processing.
    required_cols = {"name", "bio", "niche", "location", "audience_size"}
    validate_dataset(creators_df, required_cols)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Once dataset is loaded, show text area for campaign brief.
st.markdown("### Enter your campaign brief below:")
campaign_brief = st.text_area("Campaign Brief", placeholder="e.g. Looking for Indian influencers who focus on eco-friendly fashion and modern lifestyle.")

if campaign_brief.strip() == "":
    st.info("Please enter a campaign brief to find matches.")
    st.stop()

# Load the embedding model (BAAI/bge-m3) with standard pip installation.
with st.spinner("Loading embedding model..."):
    try:
        model = SentenceTransformer("BAAI/bge-m3")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Embed creator bios (for each bio, no prompt needed).
with st.spinner("Embedding creator bios..."):
    try:
        bio_texts = creators_df["bio"].tolist()
        bio_embeddings = model.encode(bio_texts, convert_to_tensor=False, normalize_embeddings=True)
        creators_df["bio_embedding"] = bio_embeddings.tolist()  # store row-wise as list of floats
    except Exception as e:
        st.error(f"Error embedding bios: {e}")
        st.stop()

# Embed campaign brief with BGE prompt (recommended prefix for search queries).
with st.spinner("Embedding campaign brief..."):
    try:
        query = "Represent this sentence for searching relevant passages: " + campaign_brief
        query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    except Exception as e:
        st.error(f"Error embedding campaign brief: {e}")
        st.stop()

# Calculate cosine similarity between the campaign brief and creator bios.
with st.spinner("Calculating similarity scores..."):
    try:
        # Stack creator embeddings into one tensor.
        device = query_embedding.device
        creator_embeddings_tensor = torch.tensor(creators_df["bio_embedding"].tolist(), device=device)
        similarities = util.cos_sim(query_embedding, creator_embeddings_tensor)[0].cpu().numpy()
        creators_df["similarity_score"] = similarities
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        st.stop()

# Sidebar filters: Allow filtering based on niche and location.
st.sidebar.header("🔧 Filter Results")
unique_niches = sorted(creators_df["niche"].unique())
unique_locations = sorted(creators_df["location"].unique())

selected_niche = st.sidebar.selectbox("Filter by Niche", ["All"] + unique_niches)
selected_location = st.sidebar.selectbox("Filter by Location", ["All"] + unique_locations)

# Apply filters if selected.
filtered_df = creators_df.copy()
if selected_niche != "All":
    filtered_df = filtered_df[filtered_df["niche"] == selected_niche]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df["location"] == selected_location]

# Sort by similarity score and select top 10 matching creators.
top_creators = filtered_df.sort_values(by="similarity_score", ascending=False).head(10)

# Display the matching creator cards.
st.markdown("### 🎯 Top Matching Creators")
if top_creators.empty:
    st.warning("No matching creators found based on the filters. Please adjust your filters or campaign brief.")
else:
    for _, row in top_creators.iterrows():
        display_creator_card(row)

# Option to download the results as CSV.
st.markdown("### 📥 Download Results")
csv_data = top_creators.to_csv(index=False)
st.download_button(label="Download CSV", data=csv_data, file_name="top_creators.csv", mime="text/csv")

