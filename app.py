import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    model = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")
    return model

def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    
    for pred in predictions:
        box = pred["box"]
        label = pred["label"]
        score = pred["score"]
        
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)
        
        text = f"{label}: {score:.2f}"
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)
        
    return image


detector = load_model()

st.title("📸 Interactive Zero-Shot Data Labeling Agent")
st.write("Upload an image and type what you want to find, separated by commas. For example: 'a cat, a red ball, a person wearing a hat'")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
text_prompts_input = st.text_input("Enter object descriptions (comma-separated):")


if uploaded_file is not None and text_prompts_input:
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Original Image")
        st.image(uploaded_file, use_column_width=True)

    with col2:
        st.write("### Labeled Image")
        
        with st.spinner('Analyzing image...'):
            image = Image.open(uploaded_file)
            
            candidate_labels = [label.strip() for label in text_prompts_input.split(',')]
            
            if candidate_labels:
                predictions = detector(image, candidate_labels=candidate_labels)
                
                if predictions:
                    processed_image = draw_boxes(image.copy(), predictions)
                    st.image(processed_image, use_column_width=True)
                else:
                    st.write("No objects found for the given descriptions.")
            else:
                st.write("Please enter at least one object description.")