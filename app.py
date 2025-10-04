import warnings

warnings.filterwarnings("ignore")
import os,requests,openai,cohere
import gradio as gr 
from pathlib import Path 
from langchain.document_loaders import YoutubeLoader 
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Qdrant 
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
from langchain.chains.summarize import load_summarize_chain

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_CLUSTER_URL = os.environ["QDRANT_CLUSTER_URL"]
QDRANT_COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
OPEN_API_KEY = os.environ["OPENAI_API_KEY"]
prompt_file = "prompt_template.txt"

def yt_loader(yt_url):
    res = requests.get(f"https://www.youtube.com/oembed?url={yt_url}")
    if res.status_code !=200:
        yield "Invalid Youtube URL. Kindly, paste here a valid Youtube URL."
        return 
    
    yield "Extracting transcript from youtube url..."
    loader = YoutubeLoader.from_youtube_url(yt_url,add_video_info=True)
    transcript = loader.load()
    
    video_id = transcript[0].metadata["source"]
    title = transcript[0].metadata["title"]
    author = transcript[0].metadata["author"]
    
    docs = []
    for i in range(len(transcript)):
        doc = Document(page_content=transcript[i].page_content)
        docs.append(doc)
        