import streamlit as st
from llm_functions import getLLMResponse
from dotenv import load_dotenv

load_dotenv()


supported_models = {
    # 'OpenAI GPT-4':"gpt-4",
    'Llama 2 7B-Chat GGML (Quantized)': 'llama-2-7b-chat',
    'Cohere':"cohere",
    'Google PaLM (Gemini)':"google-palm",
}

st.header("Langchain Streamlit test")

# file_content = st.file_uploader("Upload PDF",type=".pdf")
# if file_content is not None:
#     print("file_content",file_content)


article_text = st.text_area("Enter article text:")

# Radio button to switch between models
model_choice = st.radio(
    label="Choose an LLM model:",
    options=list(supported_models.keys())
)


if st.button("Generate new article"):
    try:
        llm_response = getLLMResponse(article_text,model_name=supported_models[model_choice])
        if llm_response is not None:
            st.write(llm_response['text'])
        else:
            raise Exception("LLM Failed Failed")
    except Exception as e:
        st.error(e)