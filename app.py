import streamlit as st
import os
import shutil
from predict import predict
import time

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")

st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file (WAV) to verify its genre!")

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = "temp_upload"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(file_path, format='audio/wav')
    
    if st.button("Classify Genre"):
        with st.spinner("Analyzing audio..."):
            # Run prediction
            predicted_genre, confidence = predict(file_path)
            
            # Display result
            st.success(f"Prediction: **{predicted_genre.title()}**")
            st.info(f"Confidence: **{confidence:.2f}**")
            
            # Display graphs
            col1, col2 = st.columns(2)
            
            graph_dir = "graph"
            prob_plot = os.path.join(graph_dir, "prediction_probabilities.png")
            spec_plot = os.path.join(graph_dir, "spectrogram.png")
            
            # Add a small delay/check to ensure files are written (predict is synchronous but just to be safe)
            time.sleep(0.5) 
            
            with col1:
                if os.path.exists(prob_plot):
                    st.image(prob_plot, caption="Prediction Probabilities")
                else:
                    st.error("Probability graph not found.")
                    
            with col2:
                if os.path.exists(spec_plot):
                    st.image(spec_plot, caption="Spectrogram")
                else:
                    st.error("Spectrogram not found.")
