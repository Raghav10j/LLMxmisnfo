import os
from typing import Optional, Tuple

import gradio as gr
import pickle
from query_data import get_chain
from threading import Lock
from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings


# embeddings2 = HuggingFaceEmbeddings()



with open("vectorstore_covid_ll.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.load_local("vectorstore_covid", embeddings2)



def set_openai_api_key():
    """Set the API key and return the chain."""
    api_key = os.environ.get("OPENAI_API_KEY", "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")  # Retrieve the API key from the environment variable
    if api_key:
        chain = get_chain(vectorstore)
        return chain



# class ChatWrapper:

#     def __init__(self):
#         self.lock = Lock()
#     def __call__(
#         self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
#     ):
#         """Execute the chat functionality."""
#         self.lock.acquire()
#         try:
#             history = history or []
#             # If chain is None, that is because no API key was provided.
#             if chain is None:
#                 history.append((inp, "Please paste your OpenAI key to use"))
#                 return history, history
#             # Set OpenAI key
#             import openai
#             openai.api_key = api_key
#             # Run chain and append input.
#             result = chain({"question": inp, "chat_history": history})
#             output = result["answer"] + " Sources: " + result["sources"]
#             history.append((inp, output))
#         except Exception as e:
#             raise e
#         finally:
#             self.lock.release()
#         return history, history


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self,inp: str, history: Optional[Tuple[str, str]], chain
    ):
        api_key="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"
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            result = chain({"question": inp, "chat_history": history})
            output = result["answer"] 
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    api_key="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"
    with gr.Row():
        gr.Markdown("<h3><center>Check your claims and information for real or fake</center></h3>")

    chat_interface = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about a claim or news",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.HTML("Prototype")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    # state = gr.State()
    state = gr.State()
    # agent_state = gr.State()


    chatbot = set_openai_api_key()  # Set the OpenAI API key and get the chain
    

    submit.click(
        lambda message_val, state_val: chat(message_val, state_val, chatbot),
        inputs=[message, state],
        outputs=[chat_interface, state]
    )
    # submit.click(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])
    # message.submit(chat, inputs=[message, state, agent_state], outputs=[chatbot, state])


# Launch the interface
block.launch(share=True,server_port=6006).launch()
