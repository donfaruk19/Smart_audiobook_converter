import os
import sys
import time
import json
import tempfile
import streamlit as st


# --- Audio / Speech packages ---
from pydub import AudioSegment
import speech_recognition as sr

# --- TTS engines (loaded lazily inside functions where useful) ---
from gtts import gTTS  # Online voice (fast)

# --- Document packages ---
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Initialize audio_chunks as an empty list
audio_chunks = []

# ============================================================
# Page config MUST be the first Streamlit command
# ============================================================
st.set_page_config(page_title="Donfaruk19 → Smart Audiobook Converter", layout="centered")

# ============================================================
# Environment detection
# ============================================================
def running_on_streamlit_cloud():
    return "STREAMLIT_SERVER_ENABLED" in os.environ or "STREAMLIT_CLOUD" in os.environ

CLOUD_MODE = running_on_streamlit_cloud()

# ============================================================
# Helpers
# ============================================================
def chunk_text(text, max_words=1500):
    """Split text into word-limited chunks."""
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
    """Concatenate multiple audio files to a single MP3; return durations for chapter markers."""
    combined = None
    total_duration_ms = 0
    durations = []
    for file in files:
        audio = AudioSegment.from_file(file)
        dur = len(audio)
        durations.append(dur)
        total_duration_ms += dur
        combined = audio if combined is None else combined + audio
    combined.export(output_file, format="mp3")
    return output_file, durations, total_duration_ms


def ms_to_vtt(ts_ms):
    hours = ts_ms // 3600000
    rem = ts_ms % 3600000
    minutes = rem // 60000
    rem = rem % 60000
    seconds = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def write_vtt(durations, outfile="chapters.vtt"):
    start_ms = 0
    lines = ["WEBVTT", ""]
    for i, dur in enumerate(durations):
        end_ms = start_ms + dur
        lines.append(f"Chapter {i+1}")
        lines.append(f"{ms_to_vtt(start_ms)} --> {ms_to_vtt(end_ms)}")
        lines.append("")
        start_ms = end_ms
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return outfile


def write_manifest(files, durations_ms, outfile="chapters.json"):
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


def read_document(uploaded_file):
    """Read document or text file to string. Supports TXT, PDF, EPUB, DOCX (if python-docx available), ODT (basic)."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            return " ".join([page.extract_text() or "" for page in reader.pages])
        elif name.endswith(".epub"):
            book = epub.read_epub(uploaded_file)
            text_content = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text_content.append(soup.get_text(separator=" "))
            return " ".join(text_content)
        elif name.endswith(".docx"):
            try:
                import docx
            except Exception:
                st.error("DOCX support requires python-docx. Please add it to requirements.")
                return ""
            doc = docx.Document(uploaded_file)
            return " ".join([para.text for para in doc.paragraphs])
        elif name.endswith(".odt"):
            # Minimal ODT support via BeautifulSoup if content XML is provided
            # Some ODT readers won't work with simple read; recommend installing odfpy for full support.
            try:
                content = uploaded_file.read()
                soup = BeautifulSoup(content, "xml")
                return " ".join([t.get_text(" ") for t in soup.find_all("text:p")]) or ""
            except Exception as e:
                st.error(f"Failed to read ODT: {e}")
                return ""
        else:
            st.error("Unsupported document type.")
            return ""
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        return ""


def save_uploaded_audio(uploaded_audio):
    """Save uploaded audio to a temp file and return the path."""
    try:
        suffix = f".{uploaded_audio.name.split('.')[-1].lower()}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
            tmpfile.write(uploaded_audio.read())
            tmpfile.flush()
            return tmpfile.name
    except Exception as e:
        st.error(f"Failed to save uploaded audio: {e}")
        return None


def transcribe_file(path, engine_label):
    """Transcribe audio file using selected engine."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(path) as source:
            audio_data = recognizer.record(source)
            if engine_label == "Cloud speech (free)":
                try:
                    return recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    st.error("Speech not recognized.")
                except sr.RequestError as e:
                    st.error(f"Cloud STT service error: {e}")
            elif engine_label == "Offline speech (experimental)":
                try:
                    import pocketsphinx
                except Exception:
                    st.error("Offline speech requires pocketsphinx installed locally.")
                    return ""
                try:
                    return recognizer.recognize_sphinx(audio_data)
                except sr.UnknownValueError:
                    st.error("Speech not recognized (offline).")
                except Exception as e:
                    st.error(f"Offline STT error: {e}")
            else:
                st.error("Unsupported recognition engine.")
    except Exception as e:
        st.error(f"Failed to transcribe: {e}")
    return ""


SUPPORTED_TTS_LANGS = [
    # Common, safe gTTS languages
    "en", "fr", "es", "de", "it", "pt", "ru",
    "zh-CN", "zh-TW", "ja", "ko", "hi", "ar", "tr",
    "nl", "pl", "el", "sv", "ta", "te", "th", "vi"
]


def synthesize_chunk(chunk, engine_label, language, rate, volume, allow_neural):
    """Synthesize a single chunk according to engine choice. Returns filename or raises."""
    if not chunk or not chunk.strip():
        raise ValueError("Empty text chunk")

    # Online voice (fast): gTTS
    if engine_label == "Online voice (fast)":
        if language not in SUPPORTED_TTS_LANGS:
            raise ValueError(f"Language '{language}' is not supported for online voice.")
        tts = gTTS(text=chunk, lang=language)
        filename = f"chapter_{int(time.time()*1000)}.mp3"
        tts.save(filename)
        return filename

    # Offline voice (basic): pyttsx3
    elif engine_label == "Offline voice (basic)":
        try:
            import pyttsx3
        except Exception:
            raise RuntimeError("Offline voice requires pyttsx3 installed locally.")
        engine = pyttsx3.init()
        try:
            engine.setProperty('rate', int(rate))
            engine.setProperty('volume', float(volume))
        except Exception:
            # Some voices/drivers may not support all properties
            pass
        filename = f"chapter_{int(time.time()*1000)}.mp3"
        engine.save_to_file(chunk, filename)
        engine.runAndWait()
        return filename

    # Neural voice (advanced): Coqui TTS
    elif engine_label == "Neural voice (advanced)":
        if not allow_neural:
            raise RuntimeError("Neural voice is not available in cloud mode.")
        try:
            from TTS.api import TTS
        except Exception:
            raise RuntimeError("Neural voice requires Coqui TTS installed locally.")
        # Example English model; adjust if you add multilingual models
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        filename = f"chapter_{int(time.time()*1000)}.wav"
        tts.tts_to_file(text=chunk, file_path=filename)
        return filename

    else:
        raise RuntimeError("Unsupported engine choice")


# ============================================================
# UI
# ============================================================
st.title("📚 Smart Audiobook Converter")
st.caption("Developed by Donfaruk19")

# Top-level mode
mode = st.radio("Select feature", ["Text → Audio", "Audio → Text"], key="mode_radio")

# ------------------------------------------------------------
# TEXT → AUDIO
# ------------------------------------------------------------
if mode == "Text → Audio":
    option = st.radio("Text source", ["Type text", "Upload document"], key="input_radio")

    text = ""
    if option == "Type text":
        text = st.text_area("Enter text", "Hello dear, welcome back!", key="textarea")
    else:
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["txt", "pdf", "epub", "docx", "odt"],
            key="doc_uploader"
        )
        if uploaded_file:
            text = read_document(uploaded_file)

    # Engine list (clean names)
    if CLOUD_MODE:
        engine_choice = st.radio(
            "Voice type",
            ["Online voice (fast)"],
            key="engine_radio"
        )
    else:
        engine_choice = st.radio(
            "Voice type",
            ["Offline voice (basic)", "Online voice (fast)", "Neural voice (advanced)"],
            key="engine_radio"
        )

    language = st.selectbox(
        "Speech language",
        SUPPORTED_TTS_LANGS,
        index=0,
        key="language_select"
    )
    rate = st.slider("Speech rate", 100, 250, 150, key="rate_slider")
    volume = st.slider("Volume", 0.0, 1.0, 1.0, key="volume_slider")
    merge_opt = st.checkbox("Merge chapters into one audiobook", value=True, key="merge_checkbox")

    if st.button("🎙️ Convert", key="convert_button"):
        if not text or not text.strip():
            st.error("Please provide text or upload a document.")
        else:
            chunks = chunk_text(text)
            if not chunks:
                st.error("No text content found after processing.")
            else:
                st.info(f"Preparing {len(chunks)} chapters...")
                progress = st.progress(0)
                status = st.empty()

                audio_files = []
                for i, chunk in enumerate(chunks, start=1):
                    status.markdown(f"Converting chapter {i}/{len(chunks)} …")
                    try:
                        filename = synthesize_chunk(
                            chunk=chunk,
                            engine_label=engine_choice,
                            language=language,
                            rate=rate,
                            volume=volume,
                            allow_neural=not CLOUD_MODE
                        )
                        audio_files.append(filename)
                        audio_chunks.append(filename)
                    except Exception as e:
                        st.error(f"Conversion failed for chapter {i}: {e}")
                        # Skip failed chapter and continue
                    progress.progress(i / len(chunks))
                    time.sleep(0.05)

                if not audio_files:
                    st.error("No chapters were generated.")
                else:
                    status.markdown("Generating chapter markers…")
                    if merge_opt:
                        try:
                            final_file, durations, total_ms = merge_audio(audio_files, "audiobook.mp3")
                            vtt_path = write_vtt(durations, "chapters.vtt")
                            manifest_path = write_manifest(audio_files, durations, "chapters.json")

                            st.success("✅ Audiobook ready!")
                            st.audio(final_file, format="audio/mp3")
                            with open(final_file, "rb") as f:
                                st.download_button(
                                    "Download audiobook (MP3)", f,
                                    file_name="audiobook.mp3",
                                    key="download_mp3"
                                )
                            with open(vtt_path, "rb") as f:
                                st.download_button(
                                    "Download chapter markers (WebVTT)", f,
                                    file_name="chapters.vtt",
                                    key="download_vtt"
                                )
                            with open(manifest_path, "rb") as f:
                                st.download_button(
                                    "Download chapters manifest (JSON)", f,
                                    file_name="chapters.json",
                                    key="download_json"
                                )
                        except Exception as e:
                            st.error(f"Failed to merge chapters: {e}")

                    st.markdown("### Individual chapter files")
                    for idx, file in enumerate(audio_files):
                        try:
                            st.audio(file)
                            with open(file, "rb") as f:
                                st.download_button(
                                    f"Download {os.path.basename(file)}", f,
                                    file_name=os.path.basename(file),
                                    key=f"download_chapter_{idx}"
                                )
                        except Exception as e:
                            st.error(f"Failed to render/download {file}: {e}")

# ------------------------------------------------------------
# AUDIO → TEXT
# ------------------------------------------------------------
else:
    # Recording support (if streamlit-webrtc is available)
    st.subheader("Record or upload audio for transcription")

    can_record = False
    try:
        from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
        can_record = True
    except Exception:
        st.info("Recording is available locally with streamlit-webrtc. Add it to requirements to enable in cloud.")

    if can_record:
        class AudioCollector(AudioProcessorBase):
            def __init__(self):
                self.samples = []

            def recv_audio(self, frame):
                # Collect raw audio frames
                self.samples.append(frame.to_ndarray())
                return frame

        st.write("Use the Start button below to record. Stop to finalize, then download and upload for transcription.")
        webrtc_ctx = webrtc_streamer(key="speech_capture", audio_processor_factory=AudioCollector)
        # For simplicity, we provide guidance rather than automatic saving of raw frames.

    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        key="audio_uploader"
    )

    engine_stt = st.radio(
        "Recognition engine",
        ["Cloud speech (free)", "Offline speech (experimental)"],
        key="stt_radio"
    )

    if uploaded_audio and st.button("📝 Convert to text", key="convert_text_button"):
        path = save_uploaded_audio(uploaded_audio)
        if path:
            text = transcribe_file(path, engine_stt)
            if text and text.strip():
                st.success("✅ Transcription complete")
                st.text_area("Transcribed text", text, height=250, key="transcribed_text")
                st.download_button(
                    "Download transcription (TXT)",
                    text,
                    file_name="transcription.txt",
                    key="download_transcription"
                )
            else:
                st.error("No transcription result produced.")

# ============================================================
# AI Enhancements (shared across Text→Audio and Audio→Text)
# ============================================================

from openai import OpenAI

# Initialize OpenAI client once, using Streamlit secrets
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Quick connectivity check
try:
    models = client.models.list()
    st.success("✅ OpenAI client initialized successfully.")
    st.write("models available:")
    for m in models.data[:5]:
        st.write("-", m.id)
except Exception as e:
    st.error(f"❌ Client initialized but API call failed: {e}")
    
# --- Connectivity check: list models if client works ---
available_models = []
if client:
    try:
        models = client.models.list()
        available_models = [m.id for m in models.data if "gpt" in m.id]  # filter to GPT models
        st.write("✅ OpenAI client initialized. models available:")
        for m in available_models[:5]:
            st.write("-", m)
    except Exception as e:
        st.error(f"❌ Client initialized but API call failed: {e}")
else:
    st.error("❌ OpenAI client not initialized. Check API")

# --- Helper function to call responses API safely ---
def call_ai(model, prompt):
    if client is None:
        return "OpenAI client not available."
    try:
        response = client.responses.create(
            model=model,
            input=prompt
        )
        return response.output[0].content[0].text
    except Exception as e:
        return f"❌ Error: {e}"

# --- Shared AI Enhancements UI ---
st.divider()
st.subheader("✨ AI Enhancements")

# Let user choose which model to use (populate from API if available)
if available_models:
    selected_model = st.selectbox("Choose AI model", available_models, index=0)
else:
    selected_model = "gpt-4o-mini"  # fallback default

if 'text' in locals() and text and text.strip():
    if st.button("Summarize Text"):
        summary = call_ai(selected_model, f"Summarize this text in 3 sentences:\n\n{text}")
        if summary.startswith("❌") or summary.startswith("OpenAI"):
            st.error(summary)
        else:
            st.markdown("**Summary:**")
            st.write(summary)

    target_lang = st.selectbox("Choose translation language", ["French", "Spanish", "Arabic", "Hausa", "English"])
    if st.button("Translate"):
        translation = call_ai(selected_model, f"Translate this text into {target_lang}:\n\n{text}")
        if translation.startswith("❌") or translation.startswith("OpenAI"):
            st.error(translation)
        else:
            st.markdown(f"**Translation ({target_lang}):**")
            st.write(translation)

    if st.button("Extract Keywords"):
        keywords = call_ai(selected_model, f"Extract 5 key topics from this text:\n\n{text}")
        if keywords.startswith("❌") or keywords.startswith("OpenAI"):
            st.error(keywords)
        else:
            st.markdown("**Keywords:**")
            st.write(keywords)

    if st.button("Sentiment Analysis"):
        sentiment = call_ai(selected_model, f"Analyze the sentiment of this text (positive, negative, neutral):\n\n{text}")
        if sentiment.startswith("❌") or sentiment.startswith("OpenAI"):
            st.error(sentiment)
        else:
            st.markdown("**Sentiment Analysis:**")
            st.write(sentiment)

    if st.button("Generate Outline"):
        outline = call_ai(selected_model, f"Create a structured outline of this text with main points and subpoints:\n\n{text}")
        if outline.startswith("❌") or outline.startswith("OpenAI"):
            st.error(outline)
        else:
            st.markdown("**Outline:**")
            st.write(outline)

else:
    st.info("Provide text (via upload, typing, or transcription) to enable AI features.")


# ============================================================
# Merge audio and apply AI chapter titles automatically
# ============================================================

if st.button("Merge Audiobook"):
    if audio_chunks:
        try:
            merged = AudioSegment.empty()
            durations = []
            for chunk in audio_chunks:
                try:
                    audio = AudioSegment.from_file(chunk, format="mp3")
                    merged += audio
                    durations.append(len(audio))
                except Exception as e:
                    st.error(f"❌ Failed to process chunk {chunk}: {e}")

            if not durations:
                st.error("⚠️ No valid audio chunks to merge.")
            else:
                merged_file = "audiobook.mp3"
                try:
                    merged.export(merged_file, format="mp3")
                except Exception as e:
                    st.error(f"❌ Failed to export merged audio: {e}")
                    merged_file = None

                # --- AI Chapter Titles ---
                chunks = chunk_text(text, max_words=800)
                titles = generate_chapter_titles(chunks)

                # Write WebVTT with AI titles
                vtt_ai_path = write_vtt_with_titles(durations, titles, "chapters_ai.vtt")

                # Save manifest JSON
                manifest_file = "chapters_with_titles.json"
                try:
                    chapters = [{"index": i+1, "title": t, "duration_sec": d/1000} 
                                for i, (t, d) in enumerate(zip(titles, durations))]
                    with open(manifest_file, "w", encoding="utf-8") as f:
                        json.dump({"chapters": chapters}, f, indent=2)
                except Exception as e:
                    st.error(f"❌ Failed to save manifest: {e}")
                    manifest_file = None

                # --- Downloads ---
                if merged_file:
                    with open(merged_file, "rb") as f:
                        st.download_button("Download Audiobook (MP3)", f, file_name=merged_file)
                if vtt_ai_path:
                    with open(vtt_ai_path, "rb") as f:
                        st.download_button("Download AI Chapter Markers (WebVTT)", f, file_name="chapters_ai.vtt")
                if manifest_file:
                    with open(manifest_file, "rb") as f:
                        st.download_button("Download Chapters Manifest with AI Titles (JSON)", f, file_name=manifest_file)

                # --- Inline audio player ---
                if merged_file:
                    st.audio(merged_file, format="audio/mp3")
                    st.info("AI chapter markers are available in the WebVTT file for advanced players.")

                # --- Chapter list ---
                st.markdown("### 📚 Chapter List (AI Titles)")
                if manifest_file and chapters:
                    st.table(chapters)

                # --- Jump to Chapter buttons ---
                st.markdown("### 🎯 Jump to Chapter")
                start_ms = 0
                chapter_boundaries = []
                for i, (title, dur) in enumerate(zip(titles, durations), start=1):
                    end_ms = start_ms + dur
                    chapter_boundaries.append((i, title, start_ms, end_ms))
                    if st.button(f"Go to Chapter {i}: {title}"):
                        st.write(f"⏩ Suggested start time: {start_ms/1000:.1f} seconds")
                        st.info("Use this timestamp in your player to jump directly.")
                    start_ms = end_ms

                # --- Now Playing indicator ---
                st.markdown("### 🎵 Now Playing")
                current_time = st.number_input("Enter current playback time (seconds)", min_value=0.0, step=1.0)
                active_chapter = None
                for i, title, start_ms, end_ms in chapter_boundaries:
                    if start_ms/1000 <= current_time < end_ms/1000:
                        active_chapter = (i, title)
                        break

                if active_chapter:
                    st.success(f"▶ Currently playing: Chapter {active_chapter[0]} — {active_chapter[1]}")
                else:
                    st.info("Playback time not within any chapter range.")

                st.success("Audiobook merged successfully with AI chapter titles!")

        except Exception as e:
            st.error(f"Failed to merge audiobook: {e}")
