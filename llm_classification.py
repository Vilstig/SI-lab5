import torch
from transformers import pipeline , AutoModelForCausalLM , AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from encoder_utils import load_filtered_test_data

df = load_filtered_test_data()

llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained ( llm_model_name )
model = AutoModelForCausalLM.from_pretrained (
    llm_model_name,
    torch_dtype = torch.float16,
    device_map ="auto"
)

hf_pipeline = pipeline ("text-generation", model = model , tokenizer = tokenizer ,
    temperature =0.1 , do_sample = True , pad_token_id = tokenizer.eos_token_id
)

llm = HuggingFacePipeline ( pipeline = hf_pipeline )
basic_template = """ Classify the text sentiment into one of three classes:
positive, negative, neutral.
Text : { text }
Class :"""
prompt = PromptTemplate.from_template( basic_template )

llm_chain = prompt | llm
result = llm_chain.invoke ({"text": df['sentence'][0] })
print ( result )