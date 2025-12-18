📚 Smart Audiobook & Speech Converter

A free, open‑source Streamlit app that converts text ↔ audio and supports full book‑to‑audiobook conversion with chapter markers, progress tracking, and transcription.

![Open in Streamlit]((https://smartaudiobookconverter.streamlit.app/))
!Python
!License

---

✨ Features
- 🔄 Text → Audio
  - Type text or upload books (TXT, PDF, EPUB).
  - Smart chunking splits long books into chapters.
  - Choose engine: pyttsx3 (offline), gTTS (multi‑language), Coqui TTS (neural).
  - Merge chapters into one continuous audiobook (MP3).
  - Chapter markers (WebVTT + JSON manifest).

- 🔄 Audio → Text
  - Upload WAV/MP3 files.
  - Transcribe speech using Google (free online) or PocketSphinx (offline).

- 📱 Cross‑platform
  - Works in any browser on PC or phone.
  - Deployable on Streamlit Cloud for instant sharing.

---

⚙️ Installation Modes

This project supports two environments:

1. Streamlit Cloud (lightweight)
Use the default requirements.txt file.  
It installs only the packages that work reliably on Streamlit Cloud:

`bash
pip install -r requirements.txt
`

Features available:
- Text → Audio using gTTS (Google online).
- Audio → Text using Google SpeechRecognition.
- Book upload (TXT, PDF, EPUB) with chunking and merging.
- Chapter markers and progress tracking.

---

2. Local PC/Phone (full features)
For offline voices and advanced engines, use requirements-local.txt:

`bash
pip install -r requirements-local.txt
`

Features available:
- All of the above, plus:
  - pyttsx3 (offline TTS).
  - Coqui TTS (neural voices).
  - PocketSphinx (offline speech recognition).

---

🔄 Smart Detection
The app automatically detects if it’s running on Streamlit Cloud and limits engines to lightweight ones.  
Locally, you’ll see all available engines.

---

🚀 Deployment

Local
`bash
git clone https://github.com/donfaruk19/smart-audiobook-app.git
cd smart-audiobook-app
pip install -r requirements-local.txt
streamlit run app.py
`

---

📦 Requirements
- streamlit  
- pyttsx3  
- gtts  
- TTS  
- speechrecognition  
- pocketsphinx  
- PyPDF2  
- ebooklib  
- beautifulsoup4  
- pydub  

---

🚀 Roadmap
- Add search inside table of contents.  
- Dark mode UI polish.  
- Heading detection for chapter names.  
- Offline caching for uploaded books.  

---
Made with ❤️ by Donfaruk19

