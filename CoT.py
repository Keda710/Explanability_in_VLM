import streamlit as st
import torch
from PIL import Image
import os
import sys
import io
import requests
import time
import re

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

def generate_comt_prompt():
    """Generate the Chain of Medical Thought prompt"""
    prompt = """<image>

You are an expert radiologist analyzing this medical image. First, think through your analysis step-by-step (Chain of Medical Thought), and AFTER completing your analysis, provide a separate formal radiology report.

YOU MUST STRUCTURE YOUR RESPONSE EXACTLY AS FOLLOWS:

**CHAIN OF MEDICAL THOUGHT**
First, I will think through my analysis step-by-step:

1. IMAGE IDENTIFICATION
[Provide at least 3-4 sentences identifying the image type, anatomical region, projection/view, and quality]

2. SYSTEMATIC OBSERVATION
[Write at least 5-6 sentences describing ALL visible anatomical structures in detail]
- Lung fields
- Cardiac silhouette 
- Mediastinum
- Diaphragm
- Bony structures
- Soft tissues

3. DETAILED ANALYSIS
[Write at least 6-8 sentences characterizing any abnormalities in detail]
- For each abnormality, describe: size, shape, density, margins, location, distribution
- Include measurements when applicable
- Note relationship to surrounding structures
- If normal, explicitly state normality of each major structure

4. CLINICAL CORRELATION
[Write at least 4-5 sentences connecting imaging findings to potential clinical significance]
- Discuss how findings might relate to symptoms
- Consider acuity of condition (acute, chronic, subacute)
- Note severity indicators

5. DIFFERENTIAL DIAGNOSIS
[List at least 3-4 potential diagnoses with detailed reasoning for each]
- Primary diagnosis with supporting evidence
- Alternative diagnoses with reasoning
- Explain why certain diagnoses are more/less likely

**FINAL RADIOLOGY REPORT**
[After completing your thorough analysis above, write a formal, structured radiology report]

CLINICAL INFORMATION:
[Brief relevant clinical context]

TECHNIQUE:
[Type of examination performed, technical details]

FINDINGS:
[Comprehensive description of observations, at least 6-8 sentences covering all anatomical areas]

IMPRESSION:
[Clear summary of key findings and most likely diagnosis, at least 3-4 sentences]

RECOMMENDATIONS:
[Specific follow-up studies or clinical actions if warranted, at least 2-3 recommendations]

IMPORTANT: Your response MUST contain at least 500 words total with clear separation between the Chain of Medical Thought analysis and the Final Radiology Report. Each section must be fully completed with substantial detail.
"""
    return prompt

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
    """Process image and query through the model with enhanced parameters"""
    try:
        # Prepare conversation with CoMT prompt
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], generate_comt_prompt())
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

        # First attempt with higher temperature for more detailed response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,  # Higher temperature for more detailed output
                top_p=0.95,
                max_new_tokens=2000,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # Decode output
        full_output = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        # Check if output is too short, if so, try again with different parameters
        if len(full_output.split()) < 200:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.9,  # Even higher temperature
                    top_p=0.95,
                    repetition_penalty=1.2,  # Add repetition penalty
                    max_new_tokens=2000,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            full_output = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        # Parse the output to separate Chain of Medical Thought from Final Report
        comt, report = parse_output(full_output)
        return comt, report

    except Exception as e:
        return f"Error processing image: {str(e)}", None

def parse_output(output):
    """Parse the output to separate Chain of Medical Thought from Final Report"""
    # First check for the expected headers from our enhanced prompt
    if "**CHAIN OF MEDICAL THOUGHT**" in output and "**FINAL RADIOLOGY REPORT**" in output:
        # Split directly on the headers
        parts = output.split("**FINAL RADIOLOGY REPORT**")
        if len(parts) >= 2:
            comt = parts[0].replace("**CHAIN OF MEDICAL THOUGHT**", "").strip()
            report = "**FINAL RADIOLOGY REPORT**" + parts[1].strip()
            return format_comt(comt), format_report(report)

    # Look for alternative section markers
    final_report_markers = [
        "FINAL RADIOLOGY REPORT",
        "FORMAL RADIOLOGY REPORT",
        "RADIOLOGY REPORT:",
        "IMPRESSION:",
        "CLINICAL INFORMATION:"
    ]

    for marker in final_report_markers:
        pattern = re.compile(r'(?:\*\*|\n\s*|==+\s*)?(' + re.escape(marker) + r')(?:\*\*|\s*==+)?', re.IGNORECASE)
        matches = list(pattern.finditer(output))
        
        if matches:
            split_index = matches[0].start()
            comt = output[:split_index].strip()
            report = output[split_index:].strip()
            return format_comt(comt), format_report(report)

    # If all else fails: do a simple split at 60/40 ratio
    split_index = int(len(output) * 0.6)
    comt = output[:split_index].strip()
    report = output[split_index:].strip()
    
    return format_comt(comt), format_report(report)

def format_comt(comt):
    """Format and clean up the Chain of Medical Thought section"""
    # Clean up common formatting issues
    comt = re.sub(r'^\s*\*\*CHAIN OF MEDICAL THOUGHT\*\*\s*', '', comt, flags=re.IGNORECASE)
    comt = re.sub(r'^\s*CHAIN OF MEDICAL THOUGHT\s*', '', comt, flags=re.IGNORECASE)
    comt = re.sub(r'^\s*==+\s*CHAIN OF MEDICAL THOUGHT\s*==+\s*', '', comt, flags=re.IGNORECASE)
    
    # Make sure numbered sections are properly formatted
    comt = re.sub(r'([0-9])\.\s*([A-Z])', r'\1. \2', comt)
    
    # Add clear section heading
    formatted_comt = f"CHAIN OF MEDICAL THOUGHT\n{'=' * 50}\n\n{comt.strip()}"
    
    # Add extra formatting to make subsections stand out
    formatted_comt = re.sub(r'((?:^|\n)(?:1\.|IMAGE IDENTIFICATION|SYSTEMATIC OBSERVATION|DETAILED ANALYSIS|CLINICAL CORRELATION|DIFFERENTIAL DIAGNOSIS)[^\n]*)', r'\n**\1**', formatted_comt)
    
    return formatted_comt

def format_report(report):
    """Format and clean up the Final Report section"""
    # Clean up common formatting issues
    report = re.sub(r'^\s*\*\*FINAL RADIOLOGY REPORT\*\*\s*', '', report, flags=re.IGNORECASE)
    report = re.sub(r'^\s*FINAL RADIOLOGY REPORT\s*', '', report, flags=re.IGNORECASE)
    report = re.sub(r'^\s*==+\s*FINAL RADIOLOGY REPORT\s*==+\s*', '', report, flags=re.IGNORECASE)
    
    # Add clear section heading
    formatted_report = f"FINAL RADIOLOGY REPORT\n{'=' * 50}\n\n{report.strip()}"
    
    # Add extra formatting to make subsections stand out
    formatted_report = re.sub(r'((?:^|\n)(?:CLINICAL INFORMATION|TECHNIQUE|FINDINGS|IMPRESSION|RECOMMENDATIONS)[^\n]*:)', r'\n**\1**', formatted_report)
    
    # Check if important sections exist, add placeholders if missing
    if "FINDINGS:" not in formatted_report:
        formatted_report += "\n\n**FINDINGS:**\nDetailed findings described in Chain of Medical Thought section."
        
    if "IMPRESSION:" not in formatted_report:
        formatted_report += "\n\n**IMPRESSION:**\nPlease see analysis in Chain of Medical Thought section."
        
    return formatted_report

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

# Process button
if st.button("Analyze Image") and image is not None:
    with st.spinner("Analyzing image..."):
        comt, report = process_image_and_query(
            image,
            None,  # Query is now handled internally by generate_comt_prompt()
            st.session_state.tokenizer,
            st.session_state.model,
            st.session_state.image_processor
        )
        
        # Display results in expandable sections
        with st.expander("Chain of Medical Thought Analysis", expanded=True):
            st.markdown(comt)
            
        with st.expander("Final Radiology Report", expanded=True):
            st.markdown(report)