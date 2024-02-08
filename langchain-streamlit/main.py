import streamlit as st
from llm_functions import getLLMResponse

st.header("Langchain Streamlit test")
# st.write("Model: Llama-2-70b (Bedrock)")
st.write("Model: Llama-2-7b-Chat (Local)")


file_content = st.file_uploader("Upload PDF",type=".pdf")
if file_content is not None:
    print("file_content",file_content)


user_input = st.text_input("Enter question:")
if st.button("Submit"):
    try:
        llm_response = getLLMResponse(user_input)
        if llm_response is not None:
            st.write(llm_response['text'])
        else:
            raise Exception("RAG Failed")
    except Exception as e:
        st.error(e)