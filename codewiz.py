import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    model_name = "Salesforce/codegen-350M-mono"  # Lightweight CodeGen model
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    logger.info("Model and tokenizer loaded.")
    return model, tokenizer

def generate_code(prompt, model, tokenizer, max_length=150, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,  # Set pad_token_id explicitly
        temperature=temperature,  # set at around 0.2-0.3
        top_p=top_p  # Control diversity, set at around 0.7-0.8
    )
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code

# Load model and tokenizer
logger.info("Starting model loading...")
model, tokenizer = load_model()
logger.info("Model loading completed.")

# Streamlit app
st.title("CodeWiz 🔮")
user_prompt = st.text_area("Enter your prompt:")
max_length = st.slider("Max Length", 50, 500, 150)
temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
top_p = st.slider("Top-p (Nucleus Sampling)", 0.1, 1.0, 0.9)
generate_button = st.button("Generate Code")

if generate_button and user_prompt.strip():
    with st.spinner("Generating code..."):
        generated_code = generate_code(user_prompt, model, tokenizer, max_length, temperature, top_p)
        st.code(generated_code, language="python")
