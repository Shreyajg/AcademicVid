import streamlit as st
import PyPDF2
import nltk
import base64
import os
import io
import time
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import requests
import re
import fitz
import numpy as np
from gtts import gTTS
from mutagen.mp3 import MP3
from PIL import UnidentifiedImageError
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
import imageio

nltk.download("stopwords")
nltk.download("punkt")

genai.configure(api_key="YOUR_API_KEY_HERE")

input_prompt1 = """
You are a student studying for an exam and you need quick notes with a deep understanding 
of the topic with given text. Your task is to give me key points topic-wise summary of the 
following content in simple text only. The topic headings should strictly start with a ### symbol not in bold.
"""

@st.cache_data
def get_gemini_response(input, pdf_content):
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    while True:
        try:
            response = model.generate_content([input, pdf_content])
            time.sleep(5)
            return response.text
        except Exception as e:
            print("⚠️ Error, retrying:", e)
            time.sleep(30)

@st.cache_data
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="PDF Text Key Points Extractor", page_icon=":open_book:", layout="wide")

    st.sidebar.header("Upload PDF File")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.sidebar.success("PDF file uploaded successfully!")

        with st.spinner("Reading PDF..."):
            pdf_text = read_pdf(uploaded_file)
            response = get_gemini_response(input_prompt1, pdf_text)
            responses_list = response.split("###")

            headings = []
            data = {}
            for single_response in responses_list[1:]:
                heading = single_response[: single_response.find("\n")]
                content = single_response[single_response.find("\n")+1 :]
                headings.append(heading)
                data[heading] = {"heading": heading, "content": content}

            placeholders = [st.empty() for _ in range(len(headings))]
            for i, heading in enumerate(headings):
                placeholders[i].button(
                    heading,
                    key=heading,
                    on_click=lambda key=heading: information(data, key),
                )

            st.success("Processing completed!")

def increase_audio_speed(input_audio_path, output_audio_path, speed_factor):
    audio = AudioSegment.from_mp3(input_audio_path)
    faster_audio = audio.speedup(playback_speed=speed_factor)
    faster_audio.export(output_audio_path, format="mp3")

### FIXED – search with summary context
def get_custom_search_results(query, context="", search_type="image"):
    url = "https://www.googleapis.com/customsearch/v1"
    full_query = f"{query} {context}"
    params = {
        "key": "YOUR_API_KEY_HERE",
        "cx": "YOUR_CX_ID_HERE",
        "q": full_query,
        "searchType": search_type,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])
        link_list = []
        counter = 0
        for item in items:
            try:
                url = item["link"]
                link_list.append(url)
                counter += 1
                if counter >= 4:
                    break
            except UnidentifiedImageError:
                continue
        return link_list
    else:
        print("Error:", response.status_code)
        return None

### FIXED – explain images with context
def get_image_prompt(image_url, topic_context=""):
    response = requests.get(image_url)
    with open("check.jpg", "wb") as file:
        file.write(response.content)
    img = Image.open("check.jpg")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        [
            f"You are a teacher. Context from PDF summary: {topic_context}. "
            "Explain this image clearly and simply, staying consistent with the context only.",
            img,
        ],
        stream=True,
    )
    response.resolve()
    return response.text

def get_single_video(audio_path, image_path, output_path):
    audio = MP3(audio_path)
    audio_length = audio.info.length
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image = Image.open(image_path).resize((400, 400), Image.BICUBIC)
    list_of_images = [image]
    duration = audio_length / len(list_of_images)
    duration /= 1.5
    imageio.mimsave("image.gif", list_of_images, fps=1 / duration)
    video = VideoFileClip("image.gif")
    audio_clip = AudioFileClip(audio_path)
    final_video = video.with_audio(audio_clip)
    final_video.write_videofile(output_path, fps=60, codec="libx264")

def concatenate_videos(folder_path, output_file):
    video_clips = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(folder_path, file_name)
            video_clip = VideoFileClip(file_path)
            video_clips.append(video_clip)
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(output_file)

### FIXED – video generation uses summary-driven search & context
def information(data, topic):
    progress_bar = st.progress(0)
    transcript = ""
    topic_context = data[topic]["content"]

    # search images with context
    temp = get_custom_search_results(topic, topic_context)
    if not temp:
        st.error("No images found.")
        return

    c = 1
    for url in temp:
        response = requests.get(url)
        if response.status_code == 200:
            pic = Image.open(BytesIO(response.content)).convert("RGB")
            os.makedirs("images", exist_ok=True)
            pic.save(f"images/pic{c}.jpg")
            # explain image in context
            text_expl = get_image_prompt(url, topic_context)
            transcript += text_expl + " "
            # audio
            tts = gTTS(text=text_expl, lang="en", slow=False)
            tts.save("audio.mp3")
            increase_audio_speed("audio.mp3", "audio_fast.mp3", 1.3)
            get_single_video("audio_fast.mp3", f"images/pic{c}.jpg", f"videos/video{c}.mp4")
            progress_bar.progress(int((c/len(temp))*100))
            c += 1

    concatenate_videos("videos", "final_output.mp4")
    st.video("final_output.mp4")
    with st.expander("See transcript"):
        st.write(transcript)

if __name__ == "__main__":
    main()
