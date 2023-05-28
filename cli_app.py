# import pickle
# from query_data import get_chain
import os
# os.environ["OPENAI_API_KEY"] = "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"


    
# # import boto3
# # import pickle

# # # Set up the S3 client
# # s3 = boto3.client('s3')

# # # Specify the S3 bucket and file path
# # bucket_name = 'llminfo'
# # file_key = 'vectorstore_covid.pkl'

# # # Download the pickle file from S3
# # s3.download_file(bucket_name, file_key, 'vectorstore_covid_S3.pkl')

# # # Load the pickled object
# # with open('vectorstore_covid_S3.pkl', 'rb') as file:
# #     pickled_object = pickle.load(file)



# with open("vectorstore_covid_ll.pkl", "rb") as f:
#     vectorstore = pickle.load(f)
# qa_chain = get_chain(vectorstore)
# chat_history = []
# print("Chat with your docs!")
# while True:
#     print("Human:")
#     question = input()
#     result = qa_chain({"question": question, "chat_history": chat_history})
#     chat_history.append((question, result["answer"]))
#     print("AI:")
#     print(result["answer"])

    
    
from query_data import get_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os



os.environ["OPENAI_API_KEY"] = "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"

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.load_local("vectorstore_covid", embeddings)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"] + " Sources: " + result["sources"])