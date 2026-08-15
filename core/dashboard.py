# ui/dashboard.py
# Run with: streamlit run ui/dashboard.py

import streamlit as st
import torch
from core.analyzer import AIArchitectureAnalyzer
import json

st.set_page_config(
    page_title="AI Architecture Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Architecture Analyzer")
st.subtitle("Drop any AI model and get instant analysis")

# Sidebar
with st.sidebar:
    st.header("Settings")
    deep_analysis = st.checkbox("Deep Analysis", value=True)
    show_all_layers = st.checkbox("Show All Layers", value=False)

# Model input options
tab1, tab2, tab3 = st.tabs([
    "📁 Upload Model", 
    "🤗 HuggingFace", 
    "✍️ Code Input"
])

with tab1:
    uploaded = st.file_uploader(
        "Upload your model file",
        type=['pt', 'pth', 'h5', 'onnx']
    )
    
    if uploaded:
        with st.spinner("Analyzing..."):
            # Load and analyze
            model = torch.load(uploaded)
            analyzer = AIArchitectureAnalyzer()
            result = analyzer.analyze(model)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Parameters",
                    f"{result.total_parameters:,}"
                )
            with col2:
                st.metric(
                    "Model Size",
                    f"{result.model_size_mb:.1f} MB"
                )
            with col3:
                st.metric(
                    "Layers",
                    result.layer_count
                )
            with col4:
                st.metric(
                    "Issues Found",
                    len(result.issues)
                )

with tab2:
    model_name = st.text_input(
        "Enter HuggingFace model name",
        placeholder="e.g. gpt2, bert-base-uncased, llama-2-7b"
    )
    
    if st.button("Analyze") and model_name:
        with st.spinner(f"Loading {model_name}..."):
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            analyzer = AIArchitectureAnalyzer()
            result = analyzer.analyze(model)
            
            st.json(result.__dict__)

with tab3:
    code = st.text_area(
        "Paste your model code here",
        height=300,
        placeholder="class MyModel(nn.Module): ..."
    )
    
    if st.button("Analyze Code") and code:
        # Execute code safely and analyze
        namespace = {}
        exec(code, namespace)
        # Find model class and analyze
