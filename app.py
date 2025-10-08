import warnings
warnings.filterwarnings("ignore")

import os
import requests
import gradio as gr
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Qdrant
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")
PROMPT_FILE = "prompt_template.txt"

if not GEMINI_API_KEY or not COHERE_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY and COHERE_API_KEY in your environment")

qa = None  # Global QA variable

def yt_loader(yt_url):
    """Load transcript, embed, store in Qdrant, summarize."""
    # Validate URL
    res = requests.get(f"https://www.youtube.com/oembed?url={yt_url}")
    if res.status_code != 200:
        return "Invalid YouTube URL. Please provide a valid URL."

    # Extract video ID
    if "watch?v=" not in yt_url:
        return "Invalid YouTube URL format."
    video_id = yt_url.split("watch?v=")[-1].split("&")[0]

    # Fetch transcript safely
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except (NoTranscriptFound, TranscriptsDisabled):
        return "❌ Could not fetch transcript. Check if the video has subtitles."
    except Exception as e:
        return f"❌ Error fetching transcript: {e}"

    # Convert transcript to Document objects
    docs = [Document(page_content=item['text']) for item in transcript_list]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gemini-1.5",
        chunk_size=1000,
        chunk_overlap=64,
        separators=["\n\n", "\n", " "],
    )
    docs_splitter = text_splitter.split_documents(docs)

    # Embeddings
    cohere_embeddings = CohereEmbeddings(model="large", cohere_api_key=COHERE_API_KEY)

    # Qdrant vector store
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

    # Setup QA
    global qa
    llm = ChatGoogleGenerativeAI(model_name="gemini-1.5", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Summarize
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summarize_text = chain.run(docs_splitter)

    return (
        f"Video ID: {video_id}\n"
        f"Summarized Text:\n{summarize_text}"
    )

def chat(chat_history, query):
    if qa is None:
        return chat_history + [("System", "Please load a YouTube video first.")]
    res = qa.run(query)
    progressive_response = ""
    for ele in "".join(res):
        progressive_response += ele
    return chat_history + [(query, progressive_response)]

# Gradio UI
with gr.Blocks() as demo:
    gr.HTML("<h1>🎬 YouTube AI Assistant</h1>")
    gr.Markdown(
        "Paste a YouTube URL to extract and summarize the transcript, "
        "then ask questions using Cohere + Gemini + Qdrant."
    )

    with gr.Tab("Load/Summarize YouTube Video"):
        text_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        text_output = gr.Textbox(label="Summarized Transcript")
        text_button = gr.Button(value="Build AI Bot!")
        text_button.click(yt_loader, inputs=text_input, outputs=text_output)

    with gr.Tab("AI Assistant"):
        chatbot = gr.Chatbot()
        query = gr.Textbox(label="Ask a question about the video")
        chat_button = gr.Button(value="Submit Query")
        clear_button = gr.Button(value="Clear Chat History")

        query.submit(chat, [chatbot, query], chatbot)
        chat_button.click(chat, [chatbot, query], chatbot)
        clear_button.click(lambda: None, None, chatbot)

demo.queue().launch()
