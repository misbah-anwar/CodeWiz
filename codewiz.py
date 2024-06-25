import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model():
    model_name = "EleutherAI/gpt-neo-2.7B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_code(prompt, model, tokenizer, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code

model, tokenizer = load_model()

st.title("CodeWiz 🪄")
user_prompt = st.text_area("Enter your prompt:")
max_length = st.slider("Max Length", 50, 500, 150)
generate_button = st.button("Generate Code")

if generate_button and user_prompt.strip():
    with st.spinner("Generating code..."):
        generated_code = generate_code(user_prompt, model, tokenizer, max_length)
        st.code(generated_code, language="python")