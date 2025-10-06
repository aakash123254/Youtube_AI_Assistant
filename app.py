import warnings
warnings.filterwarnings("ignore")

import os
import requests
import gradio as gr

from pathlib import Path
from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings  # you can keep using Cohere embeddings, or switch to Gemini embeddings API
from langchain.vectorstores import Qdrant

# Replace OpenAI Chat + PromptTemplate + RetrievalQA imports:
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

# Import Gemini / Google GenAI integration:
from langchain_google_genai import ChatGoogleGenerativeAI  # the LLM wrapper for Gemini models
# or directly use google.generativeai, depending on your SDK preference
import google.generativeai as genai

# Load API key for Gemini / GenAI
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY / GOOGLE_API_KEY in environment")

# If using google.generativeai library, configure it
genai.configure(api_key=GEMINI_API_KEY)

# (You can keep the rest of your Qdrant / embeddings setup same as before)
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_CLUSTER_URL = os.environ["QDRANT_CLUSTER_URL"]
QDRANT_COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]

prompt_file = "prompt_template.txt"

def yt_loader(yt_url):
    res = requests.get(f"https://www.youtube.com/oembed?url={yt_url}")
    if res.status_code != 200:
        yield "Invalid Youtube URL. Kindly, paste here a valid Youtube URL."
        return

    yield "Extracting transcript from youtube url..."
    loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=True)
    transcript = loader.load()

    video_id = transcript[0].metadata["source"]
    title = transcript[0].metadata["title"]
    author = transcript[0].metadata["author"]

    docs = [Document(page_content=seg.page_content) for seg in transcript]

    yield "Splitting transcript into chunks of text..."
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",  # this is for splitting; you can keep or change
        chunk_size=1000,
        chunk_overlap=64,
        separators=["\n\n", "\n", " "],
    )
    docs_splitter = text_splitter.split_documents(docs)

    # You can either use Cohere embeddings or switch to Google embeddings (if available)
    cohere_embeddings = CohereEmbeddings(model="large", cohere_api_key=os.environ["COHERE_API_KEY"])
    yield "Uploading chunks of text into Qdrant..."
    qdrant = Qdrant.from_documents(
        docs_splitter,
        cohere_embeddings,
        url=QDRANT_CLUSTER_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    # Read prompt template
    with open(prompt_file, "r") as file:
        prompt_template = file.read()

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

    # Use Gemini / ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model_name="gemini-2.5-flash", temperature=0)

    global qa
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )

    yield "Generating summarized text from transcript..."
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summarize_text = chain.run(docs_splitter)

    res = (
        "Video ID: " + video_id + "\n"
        + "Video Title: " + title + "\n"
        + "Channel Name: " + author + "\n"
        + "Summarized Text:\n" + summarize_text
    )
    yield res

def chat(chat_history, query):
    # Use qa.run (which uses Gemini behind the scenes) to get response
    res = qa.run(query)
    progressive_response = ""
    for ele in "".join(res):
        progressive_response += ele + ""
        yield chat_history + [(query, progressive_response)]

with gr.Blocks() as demo:
    gr.HTML("<h1>Welcome to AI YouTube Assistant (using Gemini)</h1>")
    gr.Markdown(
        "Generate transcript from Youtube url, get summarized text, and ask questions (answers based only on transcript)."
    )

    with gr.Tab("Load/Summarize Youtube Video"):
        text_input = gr.Textbox(
            label="Paste a valid Youtube URL",
            placeholder="https://www.youtube.com/watch?v=…",
        )
        text_output = gr.Textbox(label="Summarized transcript of the video")
        text_button = gr.Button(value="Build AI bot!")
        text_button.click(yt_loader, text_input, text_output)

    with gr.Tab("AI Assistant"):
        chatbot = gr.Chatbot()
        query = gr.Textbox(
            label="Type your query here, then press ‘enter’ and scroll up for the response"
        )
        chat_button = gr.Button(value="Submit Query!")
        clear = gr.Button(value="Clear Chat History")
        clear.style(size="sm")

        query.submit(chat, [chatbot, query], chatbot)
        chat_button.click(chat, [chatbot, query], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch()
