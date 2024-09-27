import os
import re
import torch
from difflib import SequenceMatcher
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# base_model = LlamaForCausalLM.from_pretrained(
#      "mistralai/Mistral-7B-Instruct-v0.1",
#      load_in_8bit=True,
#      device_map='auto',
#     )
# tokenizer = LlamaTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto" )

if torch.backends.mps.is_available():
    try:
        device = torch.device("mps")
        print("Using MPS for acceleration")
    except RuntimeError:
        device = torch.device("cpu")
        print("MPS out of memory, switching to CPU")
else:
    device = torch.device("cpu")
    print("Using CPU as fallback")


pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=200,
        truncation=True,  # Explicitly enable truncation
        temperature=0.3,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
)
local_llm = HuggingFacePipeline(pipeline=pipe)

def find_columns_match(question, input_dict):
    try:
        question_list = re.split(r'\s|,|\.', question)
        for index, string2 in enumerate(question_list):
            for string1 in input_dict.get('table1_columns'):
                score = SequenceMatcher(None,string1.lower(), string2.lower()).ratio()*100
            if score > 91:
                question_list[index] = string1 + ","
        return " ".join(question_list)
        
    except:
        return question
def query_generator(tble,cols,question):

  template = """Generate a SQL query using the following table name: {Table}, and columns as a list: {Columns}, to answer the following question:
  {question}.
  
  Output Query:
  
  """
  
  prompt = PromptTemplate(template=template, input_variables=["Table","question","Columns"])
  
  response = (prompt | local_llm).invoke({"Table": tble, "question": question, "Columns": cols})

  print(response)

transaction = [
        "transaction_id",
        "transaction_amount",
        "transaction_date",
        "transaction_type",
        "transaction_status",
        "transaction_description",
        "transaction_source_account",
        "transaction_destination_account",
        "transaction_currency",
        "transaction_fee"
    ]

inputs = ["Generate an SQL query to retrieve transaction_amount, transaction_date, transaction_type,transaction_description where transaction_id is 10",
             "Generate an SQL query to retrieve transaction_id, transaction_date, transaction_type,transaction_source_account where transaction_status is 'completed'",
             "Generate an SQL query to retrieve count of the transaction_type and their average transaction_amount, ordered by transaction_type.",
             "Generate an SQL query to retrieve list of the total transaction amount for each source account, sorted by total transaction amount in descending order.",
             "Generate an SQL query to retrieve find the maximum transaction amount for each transaction type, ordered by transaction type."]

for input in inputs:
    query_generator("transaction",transaction ,question=find_columns_match(input,transaction))