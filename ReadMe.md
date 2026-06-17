AcademicVid 📚🎥

check it out here:
https://academicvid-43ncy5utis6kwfdybwpubb.streamlit.app/

AcademicVid is an AI-powered educational video generation platform that transforms textbook PDFs into short topic-wise learning videos. The system automatically extracts content from textbooks, generates concise summaries, finds relevant visuals, creates narrated explanations, and compiles them into educational videos suitable for students.

🚀 Features
📄 PDF Processing
Upload textbook PDFs through a Streamlit interface.
Extracts text content using PyPDF2.
Extracts images and figures from PDFs using PyMuPDF (fitz).
🤖 AI-Powered Summarization
Uses Google Gemini to:
Analyze textbook content.
Generate topic-wise summaries.
Extract key learning points.
Create concise educational explanations.
🖼️ Intelligent Visual Generation
Searches for relevant educational images using Google Custom Search.
Generates topic-specific diagrams when suitable visuals cannot be found.
Falls back to dynamically generated text slides if no relevant image exists.
🎙️ Educational Narration
Uses Google Cloud Text-to-Speech.
Supports Indian English voices for improved accessibility.
Adjustable speech rate for classroom-friendly narration.
Generates audio explanations for every learning point.
🎥 Automated Video Creation
Creates short video segments for each key concept.
Synchronizes narration with visuals.
Combines multiple segments into a complete educational video.
Displays generated videos directly in the Streamlit interface

1. Clone Repository
git clone https://github.com/Shreyajg/AcademicVid.git
cd AcademicVid
2. Create Virtual Environment
python -m venv venv

Activate:

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
🔑 API Configuration

Create a .env file:

GOOGLE_API_KEY=your_gemini_api_key
CUSTOM_SEARCH_CX=your_custom_search_engine_id
Google Cloud Text-to-Speech

Download your Service Account JSON key and set:

Windows:

setx GOOGLE_APPLICATION_CREDENTIALS "D:\path\tts_key.json"

Linux/Mac:

export GOOGLE_APPLICATION_CREDENTIALS="/path/tts_key.json"

Restart your terminal or IDE after setting the variable.

▶️ Running the Application
streamlit run app.py

Open:

http://localhost:8501
📖 Usage
Upload a textbook PDF.
Wait for topic extraction and summarization.
Select a topic from the generated list.
AcademicVid:
Extracts key points.
Finds or generates visuals.
Creates narration.
Generates video segments.
Produces a final educational video.
Watch the generated video and review the transcript.

🎯 Target Audience
School students
Government school learners
Self-paced learners
Teachers creating supplementary content
Educational institutions

🔮 Future Improvements
Multi-language narration (English + Hindi + Kannada)
Subtitle generation
Interactive quizzes after videos
Chapter-wise revision mode
Offline deployment support
Personalized difficulty levels
Animated educational diagrams
Learning analytics dashboard
👨‍💻 Authors

Developed as an AI-powered educational content generation project to improve accessibility and simplify textbook learning through automated video creation.

📜 License

This project is intended for educational and research purposes.
