import sys
import os
import shutil
import subprocess

# ============================================================
# Environment detection
# ============================================================
def running_on_streamlit_cloud():
    keys = os.environ.keys()
    return (
        "STREAMLIT_SERVER_RUN_ON_SAVE" in keys
        or "STREAMLIT_RUNNER_FAST_RERUNS" in keys
        or "SF_PARTNER" in keys
    )

CLOUD_MODE = running_on_streamlit_cloud()
print("🔎 CLOUD_MODE =", CLOUD_MODE)

# ============================================================
# Local Python version bootstrap (skip on cloud)
# ============================================================
def restart_with_python311():
    py311 = shutil.which("python3.11")
    if py311:
        print("🔧 Using Python 3.11 for full feature support...")
        os.execv(py311, [py311, "app.py", "--skip-launcher"])
    else:
        print("⚠️ Python 3.11 not found. Running with system python3 (fallback mode).")

if not CLOUD_MODE and "--skip-launcher" not in sys.argv:
    if not sys.version.startswith("3.11"):
        restart_with_python311()

# ============================================================
# Local environment setup (skip on cloud)
# ============================================================
def ensure_local_env():
    if CLOUD_MODE:
        print("☁️ Cloud mode detected — skipping local venv setup.")
        return

    venv_path = os.path.join(os.getcwd(), "venv")
    python_in_venv = os.path.join(venv_path, "bin", "python")

    # Step 1: Create venv if missing
    if not os.path.exists(venv_path):
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created.")

    # Step 2: Install local dependencies
    deps_file = "requirements-local.txt"
    if os.path.exists(deps_file):
        with open(deps_file) as f:
            deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # Skip Coqui TTS if Python >= 3.12
        if sys.version_info >= (3, 12):
            deps = [d for d in deps if not d.lower().startswith("coqui-tts")]
            print("⚠️ Skipping Coqui TTS (not supported on Python 3.12+)")

        subprocess.run([python_in_venv, "-m", "pip", "install"] + deps, check=True)
        print("✅ Local dependencies installed.")

    # Step 3: Restart inside venv if not already there
    if sys.executable != python_in_venv:
        print("🔄 Restarting inside virtual environment...")
        os.execv(python_in_venv, [python_in_venv] + sys.argv)

if not CLOUD_MODE:
    ensure_local_env()
else:
    print("☁️ Cloud mode detected — skipping ensure_local_env()")

# --- other imports ---
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

    # Detect if Coqui TTS is available locally
    def coqui_tts_available():
        try:
            import importlib.util
            return importlib.util.find_spec("TTS") is not None
        except Exception:
            return False

    # Engine list (adaptive)
    if CLOUD_MODE:
        # Cloud mode: only online voice
        engine_choice = st.radio(
            "Voice type",
            ["Online voice (fast)"],
            key="engine_radio"
        )
    else:
        # Local mode
        options = ["Offline voice (basic)", "Online voice (fast)"]
        if coqui_tts_available() and sys.version_info < (3, 12):
            options.append("Neural voice (advanced)")
        engine_choice = st.radio(
            "Voice type",
            options,
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

                    st.markdown("Individual chapter files")
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

# = = [{"index": i+1, "title": t, "duration_sec": d/1000} 
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


HF_API_KEY = st.secrets["huggingface"]["api_key"]
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def call_ai(task, text, target_lang=None):
    try:
        if task == "summarize":
            url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            resp = requests.post(url, headers=HF_HEADERS, json={"inputs": text})
            return resp.json()[0]["summary_text"]

        elif task == "translate":
            model_map = {
                "French": "Helsinki-NLP/opus-mt-en-fr",
                "Spanish": "Helsinki-NLP/opus-mt-en-es",
                "Arabic": "Helsinki-NLP/opus-mt-en-ar",
                "Hausa": "Helsinki-NLP/opus-mt-en-ha",
                "English": "Helsinki-NLP/opus-mt-ha-en"
            }
            model = model_map.get(target_lang, "Helsinki-NLP/opus-mt-en-fr")
            url = f"https://api-inference.huggingface.co/models/{model}"
            resp = requests.post(url, headers=HF_HEADERS, json={"inputs": text})
            return resp.json()[0]["translation_text"]

        elif task == "keywords":
            url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            payload = {"inputs": text,
                       "parameters": {"candidate_labels": ["politics","economy","health","education","technology","culture","sports"]}}
            resp = requests.post(url, headers=HF_HEADERS, json=payload)
            return ", ".join(resp.json()["labels"][:5])

        elif task == "sentiment":
            url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
            resp = requests.post(url, headers=HF_HEADERS, json={"inputs": text})
            return resp.json()[0]["label"]

        elif task == "outline":
            sentences = text.split(". ")
            return "\n".join([f"- {s.strip()}" for s in sentences if s.strip()])

        elif task == "ner":
            url = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
            resp = requests.post(url, headers=HF_HEADERS, json={"inputs": text})
            entities = resp.json()[0]["entities"]
            return ", ".join([f"{e['word']} ({e['entity']})" for e in entities])

        elif task == "qa":
            url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
            resp = requests.post(url, headers=HF_HEADERS,
                                 json={"inputs": {"question": "What is the main topic?", "context": text}})
            return resp.json()["answer"]

        else:
            return "❌ Unsupported task"
    except Exception as e:
        return f"❌ Error calling Hugging Face API: {e}"

def generate_chapter_titles(chunks):
    titles = []
    for chunk in chunks:
        try:
            url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            resp = requests.post(url, headers=HF_HEADERS, json={"inputs": chunk})
            titles.append(resp.json()[0]["summary_text"])
        except Exception:
            titles.append(chunk[:50] + "...")
    return titles

def write_vtt_with_titles(durations, titles, outfile="chapters_ai.vtt"):
    start_ms = 0
    lines = ["WEBVTT", ""]
    for i, (dur, title) in enumerate(zip(durations, titles)):
        end_ms = start_ms + dur
        lines.append(f"Chapter {i+1}: {title}")
        lines.append(f"{ms_to_vtt(start_ms)} --> {ms_to_vtt(end_ms)}")
        lines.append("")
        start_ms = end_ms
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return outfile

# ============================================================
# Unified AI Enhancements + Merge Audiobook Section with Dropdown
# ============================================================
st.divider()
st.subheader("✨ AI Enhancements & Audiobook Tools (Hugging Face)")

if 'text' in locals() and text and text.strip():
    feature = st.selectbox("Choose an AI feature", 
                           ["Summarize", "Translate", "Extract Keywords", "Sentiment Analysis",
                            "Generate Outline", "Named Entity Recognition", "Question Answering", "Merge Audiobook"])

    if st.button("Run Selected Feature"):
        if feature == "Summarize":
            st.write(call_ai("summarize", text))
        elif feature == "Translate":
            target_lang = st.selectbox("Choose translation language", ["French", "Spanish", "Arabic", "Hausa", "English"])
            st.write(call_ai("translate", text, target_lang))
        elif feature == "Extract Keywords":
            st.write(call_ai("keywords", text))
        elif feature == "Sentiment Analysis":
            st.write(call_ai("sentiment", text))
        elif feature == "Generate Outline":
            st.write(call_ai("outline", text))
        elif feature == "Named Entity Recognition":
            st.write(call_ai("ner", text))
        elif feature == "Question Answering":
            st.write(call_ai("qa", text))
        elif feature == "Merge Audiobook":
            if audio_chunks:
                try:
                    merged = AudioSegment.empty()
                    durations = []
                    for chunk in audio_chunks:
                        audio = AudioSegment.from_file(chunk, format="mp3")
                        merged += audio
                        durations.append(len(audio))

                    merged_file = "audiobook.mp3"
                    merged.export(merged_file, format="mp3")

                    chunks = chunk_text(text, max_words=800)
                    titles = generate_chapter_titles(chunks)
                    vtt_ai_path = write_vtt_with_titles(durations, titles, "chapters_ai.vtt")

                    manifest_file = "chapters_with_titles.json"
                    chapters = [{"index": i+1, "title": t, "duration_sec": d/1000} 
                                for i, (t, d) in enumerate(zip(titles, durations))]
                    with open(manifest_file, "w", encoding="utf-8") as f:
                        json.dump({"chapters": chapters}, f, indent=2)

                    with open(merged_file, "rb") as f:
                        st.download_button("Download Audiobook (MP3)", f, file_name=merged_file)
                    with open(vtt_ai_path, "rb") as f:
                        st.download_button("Download AI Chapter Markers (WebVTT)", f, file_name="chapters_ai.vtt")
                    with open(manifest_file, "rb") as f:
                        st.download_button("Download Chapters Manifest (JSON)", f, file_name=manifest_file)

                    st.audio(merged_file, format="audio/mp3")
                    st.table(chapters)

                except Exception as e:
                    st.error(f"Failed to merge audiobook: {e}")
else:
    st.info("Provide text (via upload, typing, or transcription) to enable AI features.")
