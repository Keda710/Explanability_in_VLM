import streamlit as st
import torch
from PIL import Image
import os
import sys
import io
import requests

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.image_processor = None
    st.session_state.context_len = None

def load_llava_model():
    """Load the LLaVA model and return components"""
    try:
        # Disable CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        disable_torch_init()

        # Set model paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "llava-rad")
        model_base = os.path.join(base_dir, "models", "vicuna-7b-v1.5")
        
        # Load model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name="llavarad",
            load_8bit=False,
            load_4bit=False,
            device_map="cpu",
            device="cpu"
        )
        return tokenizer, model, image_processor, context_len
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def process_image_and_query(image, query, tokenizer, model, image_processor):
    """Process image and query through the model"""
    try:
        # Prepare conversation
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], f"<image>\n{query}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process image
        if isinstance(image, str):  # URL
            response = requests.get(image)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].float().unsqueeze(0)

        # Generate tokens
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        stopping_criteria = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # Decode output
        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        return outputs.strip()
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Streamlit UI
st.title("LLaVA-Rad: Medical Image Analysis")
st.write("Upload a medical image or provide a URL to analyze it.")

# Load model on first run
if st.session_state.model is None:
    with st.spinner("Loading LLaVA-Rad model... This may take a few minutes..."):
        tokenizer, model, image_processor, context_len = load_llava_model()
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.image_processor = image_processor
        st.session_state.context_len = context_len
    st.success("Model loaded successfully!")

# Image input
image_source = st.radio("Select image source:", ["Upload", "URL"])
image = None

if image_source == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
else:
    url = st.text_input("Enter image URL:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        except:
            st.error("Error loading image from URL")

# Display image if available
if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Query input
query = st.text_input("Enter your query:", value="Describe the findings of the chest x-ray.")

# Process button
if st.button("Analyze Image") and image is not None:
    with st.spinner("Analyzing image..."):
        result = process_image_and_query(
            image,
            query,
            st.session_state.tokenizer,
            st.session_state.model,
            st.session_state.image_processor
        )
        st.write("### Analysis Results:")
        st.write(result)