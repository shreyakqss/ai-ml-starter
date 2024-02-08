# from langchain_community.llms.bedrock import Bedrock
from langchain.llms.ctransformers import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def initLLM():
    # llm = Bedrock(
    #     credentials_profile_name="bedrock-admin", model_id="amazon.titan-text-express-v1"
    # )

    # load a quantized version of LLama2 on CPU
    # main link: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
    # llama2 quantized models comparison:nhttps://www.reddit.com/r/LocalLLaMA/comments/14gjz8h/i_have_multiple_doubts_about_kquant_models_and/
    llm = CTransformers(model = "../model/llama-2-7b-chat.ggmlv3.q5_K_S.bin",
        model_type = "llama",
        max_new_tokens=512,
        temperature=0.2
        
        )

    return llm


# Perform RAG with LLM
def ragProcess():
    return None 

def getLLMResponse(query: str):
    llm = initLLM()
    prompt_template = """
    You are an expert article writer for a wide range of topics.
    Given this article text: {query}

    Write a short summary article of 500 words max on a similar topic but different from the given article query.
    New article content: 
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['query'])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_response = llm_chain.invoke({"query": query})
    return llm_response