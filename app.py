import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# 1. Page Config - must be the first Streamlit command
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 18px;
        color: #64748B;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .result-normal {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
    }
    .result-bacterial {
        background-color: #FFF7ED;
        border: 2px solid #F97316;
        color: #9A3412;
    }
    .result-viral {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
        letter-spacing: 1px;
    }
    .img-container {
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        background-color: #F8FAFC;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
    }
    .section-title {
        text-align: center;
        color: #334155;
        font-weight: 600;
        margin-bottom: 15px;
        font-size: 20px;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper function to convert image for HTML rendering
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Cache model loading to prevent reloading on every run
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("best_hybrid_vgg_densenet.h5")
        return model
    except Exception as e:
        return None

model = load_model()

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🩺 App Info</h2>", unsafe_allow_html=True)
    st.info("""
    **Pneumonia Detection System**  
    This tool uses a Deep Learning model (Hybrid VGG-DenseNet) to analyze chest X-ray images and detect signs of pneumonia.
    """)
    st.markdown("---")
    st.header("📝 Instructions")
    st.markdown("""
    1. **Upload** a chest X-ray image (JPG, JPEG, PNG).
    2. Wait for the model to process the image.
    3. **View the prediction** (Normal, Bacterial, or Viral) and confidence score.
    """)
    st.markdown("---")
    st.caption("Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.")

# Main Body
st.markdown("<div class='main-header'>Pneumonia Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-Powered Chest X-Ray Analysis Dashboard</div>", unsafe_allow_html=True)

# Container for upload section
with st.container():
    uploaded_file = st.file_uploader("Drop a chest X-ray image here (JPG, PNG)", type=["jpg", "jpeg", "png"])

st.write("") # Spacing

if uploaded_file is not None:
    # 2 Columns Layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='section-title'>📸 Uploaded Image</div>", unsafe_allow_html=True)
        img = Image.open(uploaded_file).convert('RGB')
        img_b64 = image_to_base64(img)
        # Display uploaded image inside a styled container
        st.markdown(f"""
        <div class='img-container'>
            <img src='data:image/jpeg;base64,{img_b64}' style='width: 100%; border-radius: 8px;'/>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>📊 Analysis Result</div>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing image..."):
            if model is None:
                st.error("Model could not be loaded. Prediction unavailable. Please ensure 'best_hybrid_vgg_densenet.h5' is in the root directory.")
            else:
                # Preprocess the image
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                
                # Normalize the image before prediction (0-1 scale)
                img_array = img_array.astype('float32') / 255.0
                
                # Expand dimensions to create a batch of 1
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Run prediction
                prediction = model.predict(img_batch)
                
                # Logic copied from original app
                if prediction.shape[-1] >= 3:
                    class_idx = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_idx])
                    classes = ["Normal", "Bacterial", "Viral"]
                    if class_idx < len(classes):
                        result = classes[class_idx]
                    else:
                        result = f"Class {class_idx}"
                elif prediction.shape[-1] == 2:
                    class_idx = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_idx])
                    result = "Pneumonia" if class_idx == 1 else "Normal"
                else:
                    prob = float(prediction[0][0])
                    if prob >= 0.5:
                        result = "Pneumonia"
                        confidence = prob
                    else:
                        result = "Normal"
                        confidence = 1.0 - prob
                
                # Determine styling based on class
                result_upper = result.upper()
                if "NORMAL" in result_upper:
                    box_class = "result-normal"
                    icon = "✅"
                    bar_color = "#10B981"  # Emerald 500
                elif "BACTERIAL" in result_upper:
                    box_class = "result-bacterial"
                    icon = "🦠" 
                    bar_color = "#F97316"  # Orange 500
                elif "VIRAL" in result_upper:
                    box_class = "result-viral"
                    icon = "⚠️"
                    bar_color = "#EF4444"  # Red 500
                else:
                    # Fallback (e.g. general 'Pneumonia')
                    box_class = "result-viral"
                    icon = "⚠️"
                    bar_color = "#EF4444"

                # Render result box inside a styled highlight box
                st.markdown(f"""
                <div class='result-box {box_class}'>
                    <div style='font-size: 48px; margin-bottom: 5px; line-height: 1;'>{icon}</div>
                    <div style='font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8;'>Prediction</div>
                    <div class='metric-value'>{result_upper}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence score with custom progress bar
                conf_percentage = confidence * 100
                st.markdown(f"""
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: 600; color: #334155;">Confidence Score</span>
                        <span style="font-weight: 700; color: {bar_color};">{conf_percentage:.1f}%</span>
                    </div>
                    <div style="width: 100%; background-color: #E2E8F0; border-radius: 9999px; height: 16px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);">
                        <div style="width: {conf_percentage}%; background-color: {bar_color}; height: 100%; 
                                    border-radius: 9999px; transition: width 1s ease-in-out;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
else:
    # If no file is uploaded, show placeholder instructions
    st.info("👆 Please upload a chest X-ray image using the uploader above to begin the analysis.")
