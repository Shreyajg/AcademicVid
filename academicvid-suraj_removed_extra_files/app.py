import streamlit as st
import PyPDF2
from rake_nltk import Metric, Rake
import nltk
from io import StringIO
import base64
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import base64
import os
import io
import time
from PIL import Image
from io import BytesIO
import pdf2image
import google.generativeai as genai
import requests
import re
import urllib.request
import fitz
import numpy as np
import re
from gtts import gTTS

# from playsound import playsound
import os
from mutagen.mp3 import MP3
from PIL import Image
import imageio
from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
import os
from pydub import AudioSegment
from PIL import UnidentifiedImageError

nltk.download("stopwords")
nltk.download("punkt")   

genai.configure(api_key="AIzaSyCbS81ajZfpFKYU2SSVb4nCDSZD5it6LO4")
input_prompt1 = """
You are a student studying for an exam and you need quick notes with a deep understanding of the topic with given text, your task is to give me key points topic-wise summary of the following content in simple text only. The topic headings should strictly start with a ### symbol not in bold.
"""


@st.cache_data

def get_gemini_response(input, pdf_content):
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content([input, pdf_content])
    return response.text


@st.cache_data
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def main():
    st.set_page_config(
        page_title="PDF Text Key Points Extractor",
        page_icon=":open_book:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    

    st.sidebar.header("Upload PDF File")
    st.subheader("Topics:")

    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.sidebar.success("PDF file uploaded successfully!")

        # Read PDF and display text
        with st.spinner("Reading PDF..."):
            time.sleep(5)
            pdf_text = read_pdf(uploaded_file)
            response = get_gemini_response(input_prompt1, pdf_text)
            print(response)
            # old one
            # res = find_keywords(pdf_text)
            # st.text_area("Extracted Text", response, height=400)
            # print("Responses"+response)

            ####trying run all in one stretch
            responses_list = response.split("###")

            # Display extracted image from pdf
            all_images = []
            all_images = extract_images_from_pdf(uploaded_file, all_images)
            # print(all_images)
            headings = []
            for response in responses_list[1:]:
                headings.append(response[: response.find("\n")])
            print(headings)
            imagemap = {}
            for image_byte in all_images:
                map_image_heading(imagemap, image_byte, headings)
            print("map :", imagemap.keys())

            # display headings
            # Display headings
            placeholders = [st.empty() for _ in range(len(responses_list) - 1)]

            data = {}
            for i, single_response in enumerate(responses_list[1:]):
                response_dic = {}
                response_dic["heading"] = single_response[: single_response.find("\n")]
                response_dic["content"] = single_response[
                    single_response.find("\n") + 1 :
                ]
                placeholders[i].button(
                    response_dic["heading"],
                    key=response_dic["heading"],
                    on_click=lambda key=response_dic["heading"]: information(
                        data, imagemap, key
                    ),
                )
                data[response_dic["heading"]] = response_dic
            print("Data ", data)

            success_placeholder = st.empty()
            success_placeholder.success("Processing completed!")
            time.sleep(3)
            success_placeholder.empty()


# increasing the audio speed
def increase_audio_speed(input_audio_path, output_audio_path, speed_factor):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_audio_path)

    # Change the speed of the audio
    faster_audio = audio.speedup(playback_speed=speed_factor)

    # Save the modified audio
    faster_audio.export(output_audio_path, format="mp3")


# explaintation about subtopic
def information(data, imagemap, topic):
    keyword_status_placeholder = st.empty()
    video_status_placeholder = st.empty()
    progress_bar_placeholder = st.empty()
    progress = 0
    video_status_placeholder.info(f"Generating video... {progress}% .")
    
    st.markdown(
        """
    <style>
        .stProgress > div > div > div > div {
            background-color: green;
        }
    </style>""",
        unsafe_allow_html=True,
    )

    progress_bar_placeholder.progress(0)
    print("Topic ", topic)
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    if topic in imagemap:
        placeholder0.subheader(topic)
        pic1 = Image.open(io.BytesIO(imagemap[topic]))
        if not os.path.exists("images"):
            os.makedirs("images")
        pic1.save("images/pic1.jpg")
        temp = get_custom_search_results(topic)
        print(temp)
        c = 2
        for i in temp[:3]:
            response = requests.get(i)
            if response.status_code == 200:
                pic = Image.open(BytesIO(response.content))
                if not os.path.exists("images"):
                    os.makedirs("images")
                if pic.mode.endswith("A"):
                    # Convert image to RGB mode if it has an alpha channel
                    pic = pic.convert("RGB")
                else:
                    # Convert other modes to RGB
                    pic = pic.convert("RGB")
                # if pic.mode == 'RGBA' or pic.mode == 'P':
                #     pic = pic.convert('RGB')
                pic.save(f"images/pic{c}.jpg")
                c = c + 1
        progress += 10
        video_status_placeholder.info(f"Generating video... {progress}% .")
        progress_bar_placeholder.progress(10)
        my_text_original = ""
        my_text_original = clean_text(get_discription(imagemap[topic]))
        bad_chars = [";", ":", "!", "*", "#", "-"]
        for i in bad_chars:
            my_text = my_text_original.replace(i, "")
        myobj = gTTS(text=my_text, lang="en", slow=False)
        myobj.save("welcome.mp3")
        increase_audio_speed("welcome.mp3", "welcome_fast.mp3", speed_factor=1.3)
        get_single_video(f"welcome_fast.mp3", "images/pic1.jpg", "videos/video1.mp4")

        my_text_original = ""
        my_text = ""
        transcript = ""
        c = 2
        # Iterate over each URL returned by get_custom_search_results(topic)
        for url in temp[:3]:
            progress += 20
            # Generate prompt for the current URL and concatenate it to my_text
            my_text_original = get_image_prompt(url)
            bad_chars = [";", ":", "!", "*", "#", "-"]
            for i in bad_chars:
                my_text = my_text_original.replace(i, "")
                transcript += my_text
            myobj = gTTS(text=my_text, lang="en", slow=False)
            myobj.save("welcome.mp3")
            increase_audio_speed("welcome.mp3", "welcome_fast.mp3", speed_factor=1.3)
            get_single_video(
                f"welcome_fast.mp3", f"images/pic{c}.jpg", f"videos/video{c}.mp4"
            )
            video_status_placeholder.info(f"Generating video... {progress}% .")
            progress_bar_placeholder.progress(progress)
            c = c + 1

        concatenate_videos("videos", "final_output.mp4")
        placeholder1.video("final_output.mp4")
        # placeholder2.text(data[topic]["content"])
        with st.expander("See transcript"):
            st.write(transcript)

    else:
        placeholder0.subheader(topic)
        temp = get_custom_search_results(topic)
        print(temp)
        c = 1
        for i in temp:
            response = requests.get(i)
            if response.status_code == 200:
                pic = Image.open(BytesIO(response.content))
                if not os.path.exists("images"):
                    os.makedirs("images")
                if pic.mode.endswith("A"):
                    # Convert image to RGB mode if it has an alpha channel
                    pic = pic.convert("RGB")
                else:
                    # Convert other modes to RGB
                    pic = pic.convert("RGB")
                # if pic.mode == 'RGBA':
                #     pic = pic.convert('RGB')
                pic.save(f"images/pic{c}.jpg")
                c = c + 1
        progress += 10
        video_status_placeholder.info(f"Generating video... {progress}% .")
        progress_bar_placeholder.progress(progress)
        # Initialize an empty string to store concatenated prompts
        my_text_original = ""
        my_text = ""
        c = 1
        transcript = ""
        # Iterate over each URL returned by get_custom_search_results(topic)
        for url in get_custom_search_results(topic):
            progress += 20
            
            # Generate prompt for the current URL and concatenate it to my_text
            my_text_original = clean_text(get_image_prompt(url))
            bad_chars = [";", ":", "!", "*", "#", "-"]
            for i in bad_chars:
                my_text = my_text_original.replace(i, "")
                transcript += my_text
            myobj = gTTS(text=my_text, lang="en", slow=False)
            myobj.save("welcome.mp3")
            increase_audio_speed("welcome.mp3", "welcome_fast.mp3", speed_factor=1.3)
            get_single_video(
                f"welcome_fast.mp3", f"images/pic{c}.jpg", f"videos/video{c}.mp4"
            )
            c = c + 1
            video_status_placeholder.info(f"Generating video... {progress}% .")
            progress_bar_placeholder.progress(progress)

        concatenate_videos("videos", "final_output.mp4")
        placeholder1.video("final_output.mp4")
        # placeholder2.text(data[topic]["content"])
        with st.expander("See transcript"):
            st.write(transcript)
    video_status_placeholder.info(f"Generating video... 100%.")
    progress_bar_placeholder.progress(100)
    time.sleep(3)
    progress_bar_placeholder.empty()
    video_status_placeholder.empty()
    video_status_placeholder.info("Video generation completed!")
    time.sleep(3)
    video_status_placeholder.empty()


# not implemented yet must be searched here first before searching google for images. Need to crack find page number in pdf for easy use
@st.cache_data
def extract_images_from_pdf(pdf, output_array):
    pdf_data = pdf.getvalue()
    pdf_document = fitz.open(stream=pdf_data)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            if contains_figure(image):
                output_array.append(image_bytes)
    return output_array


def contains_figure(image):
    gray_image = image.convert("L")
    img_array = np.array(gray_image)
    histogram = np.histogram(img_array, bins=256, range=(0, 255))[0]
    non_zero_percentage = np.sum(histogram[1:]) / np.sum(histogram)
    threshold = 0.995
    if non_zero_percentage > threshold:
        return True
    else:
        return False


@st.cache_data
def seperate_topics(text):
    # Split the text according to topics
    topics = re.split(r"#", text)
    # Remove empty strings
    topics = [topic for topic in topics if topic]
    # Seperate out the topic and key points
    key_points = []
    for topic in topics:
        topic = topic.strip()
        topic_key_points = topic.split("\n")
        key_points.append(topic_key_points)
    return key_points


def get_custom_search_results(query, search_type="image"):
    # Define the URL
    url = "https://www.googleapis.com/customsearch/v1"

    # Define the parameters
    params = {
        "key": "AIzaSyAwnw0Hgi8kTKXSVZQ-xUyORQcRtvUntOo",
        "cx": "172574863a68442ad",
        "q": query,
        "searchType": search_type,
    }
    print(params)
    # Make the GET request
    response = requests.get(url, params=params)
    print(response)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Get the items inside the response
        items = data.get("items", [])
        link_list = []
        # for i in range(0,4):

        #     link_list.append(items[i]["link"])
        counter = 0
        for item in items:
            try:
                url = item["link"]
                get_image_prompt(url)  # Check if the image can be processed
                link_list.append(url)
                counter += 1
                if counter >= 4:
                    break  # Stop fetching URLs once 4 valid URLs are obtained
            except UnidentifiedImageError:
                continue  # Skip this URL if it's not a valid image

        return link_list
    else:
        # If the request was not successful, return None
        print("Error:", response.status_code)
        return None
import re

def clean_text(text):
    if not text:
        return ""

    # Remove LaTeX-like syntax
    text = re.sub(r"\$.*?\$", "", text)         # inline $...$
    text = re.sub(r"\\\(.*?\\\)", "", text)     # \( ... \)
    text = re.sub(r"\\\[.*?\\\]", "", text)     # \[ ... \]
    
    # Remove Markdown-like emphasis
    text = text.replace("*", "").replace("_", "")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_discription(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img.save("pdf_image.jpg")
    img = Image.open("pdf_image.jpg")
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(
        [
            "You are a teacher explaining this content to a student in class. - Keep the explanation short, clear, and educational -ask review questions at the end.  - Use simple English. - Avoid using symbols like *, /, or LaTeX unless absolutely necessary.  - Focus only on the main concept.  Now explain the following content:",
            img,
        ],
        stream=True,
    )
    response.resolve()
    cleaned=clean_text(response.text)
    print(cleaned)
    return cleaned


def get_image_prompt(image_url):
    response = requests.get(image_url)
    with open("check.jpg", "wb") as file:
        file.write(response.content)
    img = Image.open("check.jpg")
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(
        [
            "You are a teacher explaining this content to a student in class. - Keep the explanation short, clear, and educational . -ask review questions at the end - Use simple English. - Avoid using symbols like *, /, or LaTeX unless absolutely necessary.  - Focus only on the main concept.  Now explain the following content:",
            img,
        ],
        stream=True,
    )
    response.resolve()

    # ✅ clean the text
    cleaned_text = clean_text(response.text)
    print(cleaned_text)
    return cleaned_text



def map_image_heading(imagemap, image_bytes, heading):
    img = Image.open(io.BytesIO(image_bytes))
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(
        [
            "Pick one heading that suits the image omong the following headings provided dont give discription just pick one: "
            + str(",".join(heading)),
            img,
        ],
        stream=True,
    )
    response.resolve()
    imagemap[response.text] = image_bytes


def get_single_video(audio_path, image_path, output_path):
     # Load audio and get its duration
    audio = MP3(audio_path)
    audio_length = audio.info.length
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Load the single image and resize it
    image = Image.open(image_path).resize((400, 400), Image.BICUBIC)

    # Create a list with the single image
    list_of_images = [image]

    # Calculate duration for each frame
    duration = audio_length / len(list_of_images)
    duration /= 1.5

    # Save the single image as a gif
    imageio.mimsave("image.gif", list_of_images, fps=1 / duration)

    # Create video clip from the gif
    video = VideoFileClip("image.gif")

    # Load audio clip
    audio_clip = AudioFileClip(audio_path)

    # Set audio to the video clip
    final_video = video.with_audio(audio_clip)

    # Write the final video to the output path
    final_video.write_videofile(output_path, fps=60, codec="libx264")
    
def concatenate_videos(folder_path, output_file):
    video_clips = []

    # Iterate through the files in the folder
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".mp4"):  # Assuming the videos are in mp4 format
            file_path = os.path.join(folder_path, file_name)
            video_clip = VideoFileClip(file_path)
            video_clips.append(video_clip)

    # Concatenate the video clips
    final_clip = concatenate_videoclips(video_clips)

    # Write the concatenated video to a new file
    final_clip.write_videofile(output_file)


def getvideo():
    audio_path = os.path.join(os.getcwd(), "welcome_fast.mp3")
    video_path = os.path.join(os.getcwd(), ".")
    images_path = os.path.join(os.getcwd(), "images")

    audio = MP3(audio_path)
    # To get the total duration in milliseconds
    audio_length = audio.info.length

    # Get all images from the folder
    # Create a list to store all the images
    list_of_images = []
    for image_file in os.listdir(images_path):
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            image_path = os.path.join(images_path, image_file)
            image = Image.open(image_path).resize((400, 400), Image.BICUBIC)
            list_of_images.append(image)

    duration = audio_length / len(list_of_images)
    duration /= 1.5
    imageio.mimsave("images.gif", list_of_images, fps=1 / duration)

    imageio.mimsave("images.gif", list_of_images, fps=1 / duration)

    video = VideoFileClip("images.gif")
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    os.chdir(video_path)
    final_video.write_videofile(fps=60, codec="libx264", filename="video.mp4")


if __name__ == "__main__":
    main()
