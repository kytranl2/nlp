# utils.py
import re
from difflib import SequenceMatcher
from langchain.chains import LLMChain
from langchain import PromptTemplate

def find_columns_match(question, columns):
    try:
        question_list = re.split(r'\s|,|\.', question)
        for index, string2 in enumerate(question_list):
            for string1 in columns:
                score = SequenceMatcher(None, string1.lower(), string2.lower()).ratio() * 100
                if score > 91:
                    question_list[index] = string1
        return " ".join(question_list)
    except Exception as e:
        return question

def query_generator(table, columns, question, llm_chain):
    template = f"""Generate a SQL query using the following table name: {table}, and columns as a list: {columns}, to answer the following question: {question}. Output Query:"""
    prompt = PromptTemplate(template=template, input_variables=["Table", "question", "Columns"])
    response = llm_chain.run({"Table": table, "question": question, "Columns": columns})
    print(response)
