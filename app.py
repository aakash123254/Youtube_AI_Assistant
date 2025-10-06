import warnings
warnings.filterwarnings("ignore")

import os
import requests
import gradio as gr
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

# Gemini / Google GenAI chat
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "youtube_assistant")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PROMPT_FILE = "prompt_template.txt"

if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY / GOOGLE_API_KEY in environment")
if not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("Please set QDRANT_API_KEY and QDRANT_URL in environment")
if not COHERE_API_KEY:
    raise ValueError("Please set COHERE_API_KEY in environment")

# Configure Gemini chat
genai.configure(api_key=GEMINI_API_KEY)

qa = None  # global variable for RetrievalQA


def yt_loader(yt_url):
    """Load transcript, embed, store in Qdrant, summarize."""
    res = requests.get(f"https://www.youtube.com/oembed?url={yt_url}")
    if res.status_code != 200:
        return "Invalid YouTube URL. Please provide a valid URL."

    loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=True)
    transcript = loader.load()

    video_id = transcript[0].metadata["source"]
    title = transcript[0].metadata["title"]
    author = transcript[0].metadata["author"]

    docs = [Document(page_content=t.page_content) for t in transcript]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gemini-1.5",
        chunk_size=1000,
        chunk_overlap=64,
        separators=["\n\n", "\n", " "],
    )
    docs_splitter = text_splitter.split_documents(docs)

    cohere_embeddings = CohereEmbeddings(model="large", cohere_api_key=COHERE_API_KEY)
    qdrant = Qdrant.from_documents(
        docs_splitter,
        cohere_embeddings,
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME
    )

    # Load prompt template
    with open(PROMPT_FILE, "r") as f:
        prompt_template = f.read()

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )

    global qa
    llm = ChatGoogleGenerativeAI(model_name="gemini-1.5", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summarize_text = chain.run(docs_splitter)

    # Return single string for Gradio textbox
    result = (
        f"Video ID: {video_id}\n"
        f"Video Title: {title}\n"
        f"Channel Name: {author}\n"
        f"Summarized Text:\n{summarize_text}"
    )
    return result



def chat(chat_history, query):
    """Query the RetrievalQA model and return streaming response."""
    if qa is None:
        yield chat_history + [("Error", "Please load a YouTube video first!")]
        return

    res = qa.run(query)
    progressive_response = ""
    for ele in "".join(res):
        progressive_response += ele
        yield chat_history + [(query, progressive_response)]


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.HTML("<h1>Welcome to AI YouTube Assistant</h1>")
    gr.Markdown(
        "Generate transcript from a YouTube URL, get summarized text, and ask questions.\n"
        "Click 'Build AI Bot' to extract transcript and summarize.\n"
        "After summary is generated, go to 'AI Assistant' tab to ask questions about the video."
    )

    with gr.Tab("Load/Summarize YouTube Video"):
        text_input = gr.Textbox(
            label="Paste a valid YouTube URL",
            placeholder="https://www.youtube.com/watch?v=AeJ9q45PfD0"
        )
        text_output = gr.Textbox(label="Summarized transcript of the video")
        text_button = gr.Button(value="Build AI Bot!")
        text_button.click(yt_loader, text_input, text_output)

    with gr.Tab("AI Assistant"):
        chatbot = gr.Chatbot()
        query = gr.Textbox(
            label="Type your query here, then press 'enter' or click Submit"
        )
        chat_button = gr.Button(value="Submit Query!")
        clear = gr.Button(value="Clear Chat History")  # Removed .style() to fix error
        query.submit(chat, [chatbot, query], chatbot)
        chat_button.click(chat, [chatbot, query], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch()
