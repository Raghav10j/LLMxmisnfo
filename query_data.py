# from langchain.prompts.prompt import PromptTemplate
# from langchain.llms import OpenAI
# from langchain.chains import ChatVectorDBChain
import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# template = """As an AI language model, your task is to perform fact-checking based on a repository of verified and fact-checked documents. These docs are provided below as context.
# You are given the following extracted parts of a long document which fact checked and a question. Provide a conversational answer.
# Question: {question}
# =========
# {context}
# =========
# Answer:"""
# QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


# def get_chain(vectorstore):
#     llm = OpenAI(temperature=0.2)
#     qa_chain = ChatVectorDBChain.from_llm(
#         llm,
#         vectorstore,
#         qa_prompt=QA_PROMPT,
#         condense_question_prompt=CONDENSE_QUESTION_PROMPT,
#     )
#     return qa_chain


from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQAWithSourcesChain, RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# template = """As an AI language model, your task is to perform fact-checking based on a repository of verified and fact-checked documents. Use the following pieces of context to answer the users question.
# You are given the following extracted parts of a long document which fact checked and a question. Provide a conversational answer.
# Question: {question}
# =========
# {context}
# =========
# Answer:"""

system_template="""As an AI language model, your task is to perform fact-checking based on a repository of verified and fact-checked documents. Use the following pieces of context to answer the users question.
You are given the following extracted parts of a long document which is fact checked and a question. Provide an answer whether the statement is true or not based on the information from fact checked documents. Also Provide reasoning for your answer based on these factchecked documents.

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)



def get_chain(vectorstore):
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(model_name="gpt-4", temperature=0.7), chain_type="stuff",retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs)

#     chain = RetrievalQAWithSourcesChain.from_chain_type(
#     ChatOpenAI(model_name="gpt-4", temperature=0), 
#     chain_type="stuff",
#     vectorstore=vectorstore,
#     retriever=vectorstore.as_retriever(), 
#     chain_type_kwargs=chain_type_kwargs
#     )
    return chain