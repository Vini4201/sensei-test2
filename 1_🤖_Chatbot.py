import nltk
import os
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import streamlit as st

st.set_page_config(page_title="Sensei Extension",layout="wide",initial_sidebar_state="collapsed")
st.sidebar.success("Select the Required Options")

# ---------------------------------------------------frontend of chatbot-----------------
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello! Ask me anything about 🤗"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Question:", placeholder="Ask anything", key="input"
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            with st.spinner("Generating response..."):
                output = conversation_chat(
                    user_input, chain, st.session_state["history"]
                )

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="thumbs",
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="fun-emoji",
                )

class TranscriptDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


def create_conversational_chain(vector_store):
    # Create llm
    # llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
    # streaming=True,
    # callbacks=[StreamingStdOutCallbackHandler()],
    # model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
    )
    return chain


# --------------------------------------frontend of chatbots------------------------
os.environ["REPLICATE_API_TOKEN"] = "r8_K2tky5ogYIgs7td0CDwKw36FUrhWsXO3DEidA"



# Define the Streamlit app

# Set title of the browser tab

youtube_url = st.experimental_get_query_params().get("youtube_url", [""])[0]
if "youtube_url" not in st.session_state:
    st.session_state["youtube_url"]=youtube_url

# st.video(youtube_url)
# print(youtube_url)
# Set the title of the Streamlit app
st.title("Sensei Extension")
col1,col2 = st.columns([3,2])
if youtube_url:
    with col1:
        with st.expander("**PRE-REQUISITES**"):
            st.write("The output for pre-requisites here")
        # st.write(f"Embedding YouTube video from URL: {youtube_url}")

        # Use st.video to embed the YouTube video
        st.video(youtube_url)

            # Check if a YouTube URL is provided
        # Initialize Streamlit
    # uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

    # Initialize session state
    with col2:
        initialize_session_state()

        # Initialize an empty list for text_chunks
        text_chunks = []

        text = []
        # for file:
        #     file_extension = os.path.splitext(file.name)[1]
        #     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        #         temp_file.write(file.read())
        #         temp_file_path = temp_file.name

        #     loader = None
        #     if file_extension == ".pdf":
        #         loader = PyPDFLoader(temp_file_path)
        #     elif file_extension == ".docx" or file_extension == ".doc":
        #         loader = Docx2txtLoader(temp_file_path)
        #     elif file_extension == ".txt":
        #         loader = TextLoader(temp_file_path)

        #     if loader:
        #         text.extend(loader.load())
        #         os.remove(temp_file_path)

        # Ensure that text_chunks contains only strings
        text_chunks = [str(chunk) for chunk in text]

        # else:
        #     # Handle the case where no files are uploaded
        #     text_chunks = [""]  # You can provide a default or handle it differently based on your requirements

        # Wrap the transcript in a simple document class
        transcript_document = TranscriptDocument("\n".join(text_chunks))

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"},)
        # Create vector store
        vector_store = FAISS.from_documents([transcript_document], embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        # Display chat history
        display_chat_history(chain)
else:
    st.warning("No YouTube video URL provided.")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
