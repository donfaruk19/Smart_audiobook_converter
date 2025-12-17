import os
import streamlit as st
from gtts import gTTS
import tempfile
import speech_recognition as sr
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pydub import AudioSegment

# --- Detect Streamlit Cloud environment ---
def running_on_streamlit_cloud():
    return "STREAMLIT_SERVER_ENABLED" in os.environ or "STREAMLIT_CLOUD" in os.environ

CLOUD_MODE = running_on_streamlit_cloud()

# --- Helper functions ---
def chunk_text(text, max_words=1500):
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def merge_audio(files, output_file="audiobook.mp3"):
    combined = None
    durations = []
    for file in files:
        audio = AudioSegment.from_file(file)
        durations.append(len(audio))
        combined = audio if combined is None else combined + audio
    combined.export(output_file, format="mp3")
    return output_file, durations

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Audiobook Converter", layout="centered")
st.title("📚 Smart Book-to-Audiobook Converter")

mode = st.radio("Choose mode:", ["Text → Audio", "Audio → Text"], key="mode_radio")

# --- TEXT TO AUDIO ---
if mode == "Text → Audio":
    option = st.radio("Input type:", ["Type text", "Upload book"], key="input_radio")
    text = ""

    if option == "Type text":
        text = st.text_area("Enter text:", "Hello dear, welcome back!", key="textarea")
    else:
        uploaded_file = st.file_uploader("Upload a book (TXT, PDF, EPUB)", type=["txt", "pdf", "epub"])
        if uploaded_file:
            if uploaded_file.name.endswith(".txt"):
                text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
            elif uploaded_file.name.endswith(".epub"):
                book = epub.read_epub(uploaded_file)
                text_content = []
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        soup = BeautifulSoup(item.get_content(), "html.parser")
                        text_content.append(soup.get_text(separator=" "))
                text = " ".join(text_content)

    # Engine choices depend on environment
    if CLOUD_MODE:
        engine_choice = st.radio("Choose engine:", ["gTTS (Google, online)"], key="engine_radio")
    else:
        engine_choice = st.radio("Choose engine:", ["pyttsx3 (offline)", "gTTS (Google, online)", "Coqui TTS (neural)"], key="engine_radio")

    language = st.selectbox("Language", ["en", "fr", "es", "ha", "ar"], key="language_select")
    rate = st.slider("Speech rate", 100, 250, 150, key="rate_slider")
    volume = st.slider("Volume", 0.0, 1.0, 1.0, key="volume_slider")

    if st.button("🎙️ Convert to Audiobook") and text:
        chunks = chunk_text(text)
        audio_files = []
        for i, chunk in enumerate(chunks, start=1):
            if engine_choice == "pyttsx3 (offline)" and not CLOUD_MODE:
                engine = pyttsx3.init()
                engine.setProperty('rate', rate)
                engine.setProperty('volume', volume)
                filename = f"chapter_{i}.mp3"
                engine.save_to_file(chunk, filename)
                engine.runAndWait()
            elif engine_choice == "gTTS (Google, online)":
                tts = gTTS(text=chunk, lang=language)
                filename = f"chapter_{i}.mp3"
                tts.save(filename)
            elif engine_choice == "Coqui TTS (neural)" and not CLOUD_MODE:
                tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                filename = f"chapter_{i}.wav"
                tts.tts_to_file(text=chunk, file_path=filename)
            audio_files.append(filename)

        final_file, durations = merge_audio(audio_files, "audiobook.mp3")
        st.success("✅ Audiobook ready!")
        st.audio(final_file, format="audio/mp3")

# --- AUDIO TO TEXT ---
else:
    uploaded_audio = st.file_uploader("Upload audio file (WAV/MP3)", type=["wav", "mp3"])
    if uploaded_audio and st.button("📝 Convert to Text"):
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}") as tmpfile:
            tmpfile.write(uploaded_audio.read())
            tmpfile.flush()
            with sr.AudioFile(tmpfile.name) as source:
                audio_data = recognizer.record(source)
                try:
                    if CLOUD_MODE:
                        text = recognizer.recognize_google(audio_data)
                    else:
                        # Allow offline PocketSphinx locally
                        try:
                            text = recognizer.recognize_sphinx(audio_data)
                        except Exception:
                            text = recognizer.recognize_google(audio_data)
                    st.success("✅ Transcription complete:")
                    st.text_area("Transcribed text:", text, height=250)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")            

def merge_audio(files, output_file="audiobook.mp3"):
    combined = None
    total_duration_ms = 0
    durations = []
    for file in files:
        audio = AudioSegment.from_file(file)
        durations.append(len(audio))
        total_duration_ms += len(audio)
        combined = audio if combined is None else combined + audio
    combined.export(output_file, format="mp3")
    return output_file, durations, total_duration_ms

def ms_to_vtt(ts_ms):
    hours = ts_ms // (3600_000)
    rem = ts_ms % (3600_000)
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

def write_vtt(durations, outfile="chapters.vtt"):
    start_ms = 0
    lines = ["WEBVTT\n"]
    for i, dur in enumerate(durations):
        end_ms = start_ms + dur
        lines.append(f"Chapter {i+1}")
        lines.append(f"{ms_to_vtt(start_ms)} --> {ms_to_vtt(end_ms)}")
        lines.append("")  # blank line per cue
        start_ms = end_ms
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return outfile

def write_manifest(files, durations_ms, outfile="chapters.json"):
    import json
    chapters = []
    start_ms = 0
    for i, (fname, dur) in enumerate(zip(files, durations_ms)):
        chapters.append({
            "index": i + 1,
            "file": fname,
            "start_ms": start_ms,
            "duration_ms": dur
        })
        start_ms += dur
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump({"chapters": chapters}, f, indent=2)
    return outfile

# --- Modes ---
mode = st.radio("Choose mode:", ["Text → Audio", "Audio → Text"])

# --- TEXT → AUDIO ---
if mode == "Text → Audio":
    option = st.radio("Input type:", ["Type text", "Upload book"])
    text = ""

    if option == "Type text":
        text = st.text_area("Enter text:", "Hello Abdullahi, welcome back!")
    else:
        uploaded_file = st.file_uploader("Upload a book (TXT, PDF, EPUB)", type=["txt", "pdf", "epub"])
        if uploaded_file:
            if uploaded_file.name.endswith(".txt"):
                text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
            elif uploaded_file.name.endswith(".epub"):
                book = epub.read_epub(uploaded_file)
                text_content = []
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        soup = BeautifulSoup(item.get_content(), "html.parser")
                        text_content.append(soup.get_text(separator=" "))
                text = " ".join(text_content)

    engine_choice = st.radio("Choose engine:", ["pyttsx3 (offline)", "gTTS (online)", "Coqui TTS (neural, offline-capable)"])
    language = st.selectbox("Language", ["en", "fr", "es", "ha", "ar"])
    rate = st.slider("Speech rate", 100, 250, 150)
    volume = st.slider("Volume", 0.0, 1.0, 1.0)

    merge_opt = st.checkbox("Merge chapters into one audiobook", value=True)

    if st.button("🎙️ Convert"):
        if not text.strip():
            st.error("Please provide text or upload a book.")
        else:
            chunks = chunk_text(text)
            st.info(f"Preparing {len(chunks)} chapters...")
            progress = st.progress(0)
            status = st.empty()

            audio_files = []
            for i, chunk in enumerate(chunks, start=1):
                status.markdown(f"Converting chapter {i}/{len(chunks)} …")
                if engine_choice == "pyttsx3 (offline)":
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', rate)
                    engine.setProperty('volume', volume)
                    filename = f"chapter_{i}.mp3"
                    engine.save_to_file(chunk, filename)
                    engine.runAndWait()
                elif engine_choice == "gTTS (online)":
                    tts = gTTS(text=chunk, lang=language)
                    filename = f"chapter_{i}.mp3"
                    tts.save(filename)
                elif engine_choice == "Coqui TTS (neural)" and not CLOUD_MODE:
                    from TTS.api import TTS
                    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                    filename = f"chapter_{i}.wav"
                    tts.tts_to_file(text=chunk, file_path=filename)
                else:
                    st.error("X Unsupported Engine Choice")
                    

                audio_files.append(filename)
                progress.progress(i / len(chunks))
                time.sleep(0.05)

            status.markdown("Generating chapter markers…")
            if merge_opt:
                final_file, durations, total_ms = merge_audio(audio_files, "audiobook.mp3")
                vtt_path = write_vtt(durations, "chapters.vtt")
                manifest_path = write_manifest(audio_files, durations, "chapters.json")

                st.success("✅ Audiobook ready!")
                st.audio(final_file, format="audio/mp3")
                with open(final_file, "rb") as f:
                    st.download_button("Download audiobook (MP3)", f, file_name="audiobook.mp3")

                with open(vtt_path, "rb") as f:
                    st.download_button("Download chapter markers (WebVTT)", f, file_name="chapters.vtt")

                with open(manifest_path, "rb") as f:
                    st.download_button("Download chapters manifest (JSON)", f, file_name="chapters.json")

            st.markdown("### Individual chapter files")
            for file in audio_files:
                st.audio(file)
                with open(file, "rb") as f:
                    st.download_button(f"Download {os.path.basename(file)}", f, file_name=os.path.basename(file))

# --- AUDIO → TEXT ---
else:
    uploaded_audio = st.file_uploader("Upload audio file (WAV/MP3)", type=["wav", "mp3"])
    engine_stt = st.radio("Recognition engine", ["Google (free)", "PocketSphinx (offline)"])
    if uploaded_audio and st.button("📝 Convert to text"):
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}") as tmpfile:
            tmpfile.write(uploaded_audio.read())
            tmpfile.flush()
            with sr.AudioFile(tmpfile.name) as source:
                audio_data = recognizer.record(source)
                try:
                    if engine_stt == "Google (free)":
                        text = recognizer.recognize_google(audio_data)
                    else:
                        # Allow offline PocketSphinx locally
                        import pocketsphinx
                        text = recognizer.recognize_sphinx(audio_data)
                    st.success("✅ Transcription complete:")
                    st.text_area("Transcribed text:", text, height=250)
                    st.download_button("Download transcription (TXT)", text, file_name="transcription.txt")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
