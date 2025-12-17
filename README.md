# 📚 Smart Audiobook & Speech Converter

A free, open‑source Streamlit app that converts **text ↔ audio** and supports full **book‑to‑audiobook conversion** with chapter markers, progress tracking, and transcription.

## ✨ Features
- 🔄 **Text → Audio**  
  - Type text or upload books (TXT, PDF, EPUB).  
  - Smart chunking splits long books into chapters.  
  - Choose engine: `pyttsx3` (offline), `gTTS` (multi‑language), `Coqui TTS` (neural).  
  - Merge chapters into one continuous audiobook (MP3).  
  - Chapter markers (WebVTT + JSON manifest).  

- 🔄 **Audio → Text**  
  - Upload WAV/MP3 files.  
  - Transcribe speech using Google (free online) or PocketSphinx (offline).  

- 📱 **Cross‑platform**  
  - Works in any browser on PC or phone.  
  - Deployable on Streamlit Cloud for instant sharing.  

## 🛠️ Installation (Local)
```bash
git clone https://github.com/yourusername/smart-audiobook-app.git
cd smart-audiobook-app
pip install -r requirements.txt
streamlit run app.py