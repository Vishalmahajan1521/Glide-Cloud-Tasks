import streamlit as st
import requests
import json
from typing import Optional

# Configuration
API_URL = "http://127.0.0.1:8000/api/v1"

# Page config
st.set_page_config(
    page_title="AI/ML Patent Search Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI/ML Patent Search Tool")
st.markdown("---")

# Sidebar for filters
st.sidebar.header("🔧 Filters")

# Search query
query = st.text_input(
    "Enter search query:",
    placeholder="e.g., transformer architecture, medical diagnosis using ai",
    key="search_query"
)

# Number of results
top_k = st.sidebar.slider(
    "Number of results",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

# Advanced filters
st.sidebar.subheader("Advanced Filters")

jurisdictions = st.sidebar.multiselect(
    "Jurisdiction",
    options=["US", "EP", "DE", "WO"],
    key="jurisdiction_filter"
)

assignees = st.sidebar.text_input(
    "Assignee (comma-separated)",
    placeholder="e.g., Google, Microsoft, IBM",
    key="assignee_filter"
)

topic = st.sidebar.selectbox(
    "Topic",
    options=["All", "ml_healthcare", "Natural Language Processing", "Deep Learning", "Computer Vision"],  # Add more as needed
    key="topic_filter"
)

filing_year_range = st.sidebar.slider(
    "Filing Year Range",
    min_value=2010,
    max_value=2025,
    value=(2015, 2025),
    key="filing_year_filter"
)

patent_classes = st.sidebar.text_input(
    "Patent Class (comma-separated)",
    placeholder="e.g., G06N, G16H",
    key="patent_class_filter"
)

# Main search area
col1, col2 = st.columns([3, 1])

with col1:
    search_button = st.button("🔍 Search Patents", key="search_btn", use_container_width=True)

with col2:
    reset_button = st.button("🔄 Reset", key="reset_btn", use_container_width=True)

if reset_button:
    st.rerun()

# Execute search
if search_button and query:
    with st.spinner("Searching patents..."):
        try:
            # Build filters
            filters = {}
            
            if jurisdictions:
                filters["jurisdiction"] = jurisdictions
            
            if assignees:
                filters["assignee"] = [a.strip() for a in assignees.split(",")]
            
            if filing_year_range:
                filters["filing_year_from"] = filing_year_range[0]
                filters["filing_year_to"] = filing_year_range[1]
            
            if patent_classes:
                filters["patent_class"] = [c.strip() for c in patent_classes.split(",")]
            
            if topic != "All":
                filters["topic"] = topic
            
            # Prepare request
            payload = {
                "query": query,
                "top_k": top_k,
                "filters": filters if filters else None
            }
            
            # Make API request
            response = requests.post(
                f"{API_URL}/search",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                results = response.json()
                
                st.success(f"Found {len(results)} results")
                st.markdown("---")
                
                # Display results
                for idx, result in enumerate(results, 1):
                    with st.expander(
                        f"📄 {idx}. {result.get('title', 'N/A')} - Score: {result.get('score', 0):.4f}",
                        expanded=(idx == 1)
                    ):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Score", f"{result.get('score', 0):.4f}")
                            st.metric("Chunk Type", result.get("chunk_type", "N/A"))
                            st.metric("Filing Year", result.get("filing_year", "N/A"))
                        
                        with col2:
                            st.metric("Jurisdiction", result.get("jurisdiction", "N/A"))
                            st.metric("Patent ID", result.get("patent_id", "N/A"))
                            if result.get("patent_class"):
                                st.metric("Patent Class", ", ".join(result.get("patent_class")))
                        
                        st.markdown("**Assignee**")
                        st.write(result.get("assignee", "N/A"))
                        
                        st.markdown("**Patent Text**")
                        text = result.get("text", "No text available")
                        st.text_area(
                            "Text content:",
                            value=text,
                            height=200,
                            disabled=True,
                            key=f"text_{idx}"
                        )
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif search_button and not query:
    st.warning("⚠️ Please enter a search query")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Patent Search Tool | Powered by FastAPI + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
