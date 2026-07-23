import os
import json
import base64
import wave
import struct
import math
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, set_key
from openai import OpenAI
from gtts import gTTS
from moviepy import (
    ImageClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, CompositeAudioClip,
    vfx, afx
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np

load_dotenv(override=True)

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp_output")
TEMP_DIR.mkdir(exist_ok=True)

LANGUAGES = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Portuguese": "pt", "Japanese": "ja",
    "Korean": "ko", "Chinese": "zh-CN", "Arabic": "ar", "Russian": "ru",
    "Dutch": "nl", "Turkish": "tr", "Polish": "pl", "Swedish": "sv",
    "Thai": "th", "Vietnamese": "vi", "Indonesian": "id", "Czech": "cs",
    "Greek": "el", "Romanian": "ro", "Hungarian": "hu", "Finnish": "fi",
    "Danish": "da", "Norwegian": "no", "Tamil": "ta", "Telugu": "te",
    "Marathi": "mr", "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur", "Filipino": "tl",
}

FREE_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-20b:free",
]
PAID_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
]


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_script(topic, model, client):
    prompt = f"""Create a short video script about: {topic}

Return ONLY valid JSON with these fields:
- title: A catchy video title
- segments: An array of objects, each with:
  - narration: The voiceover text (1-2 sentences, conversational tone)
  - image_prompt: A detailed description for AI image generation (visual scene, no text in image)

Create 5-7 segments. Make the narration engaging and informative.
The image prompts should describe visually rich scenes that match each narration point.
Return ONLY the JSON, no markdown fences, no explanation."""

    completions = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )

    response_text = completions.choices[0].message.content.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(response_text)


def refine_script_with_chat(script, user_message, model, client):
    prompt = f"""Here is a video script in JSON format:
{json.dumps(script, indent=2)}

The user wants: {user_message}

Return the modified JSON script with the same structure. Return ONLY the JSON, no explanation."""

    completions = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )

    response_text = completions.choices[0].message.content.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(response_text)


def generate_ai_image(prompt, filename, client):
    try:
        r = client.chat.completions.create(
            model="openai/gpt-5-image-mini",
            messages=[{"role": "user", "content": f"Generate an image: {prompt}. Return only the image."}],
            max_tokens=2000,
        )
        images = getattr(r.choices[0].message, "images", None)
        if images and len(images) > 0:
            url = images[0]["image_url"]["url"]
            if url.startswith("data:"):
                b64_data = url.split(",", 1)[1]
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                return True
    except Exception:
        pass
    return False


def make_fallback_image(filename, prompt, width, height, font_path_bold):
    gradient_sets = [
        ((26, 26, 46), (83, 52, 131)),
        ((15, 52, 96), (233, 69, 96)),
        ((10, 25, 47), (42, 86, 140)),
        ((44, 22, 62), (160, 50, 80)),
        ((20, 40, 70), (100, 60, 140)),
        ((30, 15, 50), (180, 50, 70)),
        ((10, 30, 60), (60, 100, 160)),
    ]
    idx = hash(prompt) % len(gradient_sets) if prompt else 0
    c1, c2 = gradient_sets[idx]
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / height
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    font = ImageFont.truetype(font_path_bold, 48)
    words = prompt.split()[:8]
    short = " ".join(words) + ("..." if len(prompt.split()) > 8 else "")
    bbox = draw.textbbox((0, 0), short, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, height // 2 - 30), short, fill=(255, 255, 255, 180), font=font)
    img.save(filename)


def make_subtitle_image(text, width, height, font_path):
    img = Image.new("RGBA", (width, 140), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width, 140)], fill=(0, 0, 0, 160))
    font = ImageFont.truetype(font_path, 28)
    margin = 40
    max_w = width - 2 * margin
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = line + " " + w if line else w
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_w:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    line_height = 38
    total_h = len(lines) * line_height
    y = (140 - total_h) // 2
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), ln, fill=(255, 255, 255, 255), font=font)
        y += line_height
    return np.array(img)


def make_title_clip(title_str, duration, width, height, font_path_bold, font_path):
    bg = Image.new("RGB", (width, height), (10, 15, 30))
    draw = ImageDraw.Draw(bg)
    font_big = ImageFont.truetype(font_path_bold, min(52, width // 25))
    font_small = ImageFont.truetype(font_path, min(22, width // 55))
    bbox = draw.textbbox((0, 0), title_str, font=font_big)
    tw = bbox[2] - bbox[0]
    if tw > width - 100:
        font_big = ImageFont.truetype(font_path_bold, min(36, width // 35))
        bbox = draw.textbbox((0, 0), title_str, font=font_big)
        tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, height // 2 - 60), title_str, fill="white", font=font_big)
    line_y = height // 2 + 40
    draw.line([(width // 2 - 100, line_y), (width // 2 + 100, line_y)], fill=(233, 69, 96), width=3)
    sub = "AI Generated Video"
    bbox2 = draw.textbbox((0, 0), sub, font=font_small)
    sw = bbox2[2] - bbox2[0]
    draw.text(((width - sw) // 2, line_y + 20), sub, fill=(180, 180, 180), font=font_small)
    clip = ImageClip(np.array(bg)).with_duration(duration)
    clip = clip.with_effects([vfx.FadeIn(1.0), vfx.FadeOut(1.0)])
    return clip


def make_end_clip(duration, width, height, font_path_bold):
    bg = Image.new("RGB", (width, height), (10, 15, 30))
    draw = ImageDraw.Draw(bg)
    font = ImageFont.truetype(font_path_bold, min(44, width // 30))
    text = "Thank You For Watching"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, height // 2 - 40), text, fill="white", font=font)
    clip = ImageClip(np.array(bg)).with_duration(duration)
    clip = clip.with_effects([vfx.FadeIn(1.0), vfx.FadeOut(1.0)])
    return clip


def generate_bgm(duration, filename):
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    data = []
    freqs = [110, 165, 220, 330]
    amps = [0.15, 0.10, 0.08, 0.05]
    for i in range(num_samples):
        t = i / sample_rate
        val = sum(a * math.sin(2 * math.pi * f * t) for f, a in zip(freqs, amps))
        env = min(t / 2.0, 1.0) * min((duration - t) / 2.0, 1.0)
        val *= env * 0.15
        data.append(max(-1.0, min(1.0, val)))
    int_data = [int(v * 32767) for v in data]
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * len(int_data), *int_data))


def build_video(script, settings, progress_callback=None):
    client = get_client()
    width = 1280 if settings["resolution"] == "720p" else 1920
    height = 720 if settings["resolution"] == "720p" else 1080
    fps = settings["fps"]
    bgm_vol = settings["bgm_volume"]
    lang_code = settings["language"]
    fade_dur = 0.5

    font_path = "C:/Windows/Fonts/arial.ttf"
    font_path_bold = "C:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_path_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    work_dir = TEMP_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (work_dir / "audio").mkdir(parents=True)
    (work_dir / "images").mkdir(parents=True)

    segments = script["segments"]
    total_steps = len(segments) + 4
    current_step = 0

    def update(msg, progress=None):
        current_step_val = progress if progress is not None else current_step
        if progress_callback:
            progress_callback(msg, current_step_val / total_steps)

    clips = []
    for i, seg in enumerate(segments):
        update(f"Generating image for segment {i + 1}/{len(segments)}...")
        img_file = str(work_dir / "images" / f"segment{i + 1}.jpg")
        if not generate_ai_image(seg["image_prompt"], img_file, client):
            make_fallback_image(img_file, seg["image_prompt"], width, height, font_path_bold)

        update(f"Creating voiceover for segment {i + 1}/{len(segments)}...")
        audio_file = str(work_dir / "audio" / f"voiceover{i + 1}.mp3")
        tts = gTTS(text=seg["narration"], lang=lang_code, slow=False)
        tts.save(audio_file)

        audio_clip = AudioFileClip(audio_file)
        audio_clip = audio_clip.with_effects([afx.AudioFadeIn(0.3), afx.AudioFadeOut(0.3)])
        dur = audio_clip.duration

        update(f"Rendering clip {i + 1}/{len(segments)}...")
        img = Image.open(img_file).resize((width, height), Image.LANCZOS)
        img_arr = np.array(img)

        img_clip = ImageClip(img_arr).with_duration(dur)
        sub_img = make_subtitle_image(seg["narration"], width, height, font_path)
        sub_clip = ImageClip(sub_img).with_duration(dur).with_position(("center", height - 140))

        composite = CompositeVideoClip([img_clip, sub_clip], size=(width, height)).with_duration(dur)
        composite = composite.with_audio(audio_clip)
        composite = composite.with_effects([vfx.FadeIn(fade_dur), vfx.FadeOut(fade_dur)])
        clips.append(composite)

    update("Building title screen...")
    title_clip = make_title_clip(script["title"], 4, width, height, font_path_bold, font_path)
    clips.insert(0, title_clip)

    update("Building end screen...")
    end_clip = make_end_clip(3, width, height, font_path_bold)
    clips.append(end_clip)

    update("Concatenating clips...")
    final = concatenate_videoclips(clips, method="compose", padding=-fade_dur)

    update("Generating background music...")
    bgm_file = str(work_dir / "audio" / "bgm.wav")
    generate_bgm(final.duration + 2, bgm_file)
    bgm = AudioFileClip(bgm_file).with_duration(final.duration)
    bgm = bgm.with_effects([afx.AudioFadeIn(1.0), afx.AudioFadeOut(1.0)])
    bgm_low = bgm.with_volume_scaled(bgm_vol)

    voiceover = final.audio
    mixed_audio = CompositeAudioClip([voiceover, bgm_low])
    final = final.with_audio(mixed_audio)

    update("Final rendering...")
    output_file = str(work_dir / "final_video.mp4")
    final.write_videofile(
        output_file, fps=fps, codec="libx264",
        audio_codec="aac", preset="ultrafast", bitrate="3000k",
    )

    return output_file


def save_to_history(title, video_path):
    history_file = HISTORY_DIR / "history.json"
    history = []
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

    entry = {
        "title": title,
        "video_path": str(video_path),
        "timestamp": datetime.now().isoformat(),
    }
    history.insert(0, entry)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    return history


def load_history():
    history_file = HISTORY_DIR / "history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)
    return []


def init_session():
    defaults = {
        "script": None,
        "video_path": None,
        "step": 1,
        "generating": False,
        "chat_history": [],
        "history": load_history(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(
        page_title="VideoAgent - AI Video Generator",
        page_icon=":film_frames:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-title { font-size: 1rem; color: #666; margin-top: 0; }
    .step-header { font-size: 1.3rem; font-weight: 600; color: #0f3460; padding: 0.5rem 0; border-bottom: 2px solid #e94560; margin-bottom: 1rem; }
    .status-box { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0f3460; }
    </style>
    """, unsafe_allow_html=True)

    init_session()

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/video.png", width=64)
        st.markdown("## Settings")

        api_key = st.text_input(
            "API Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Your OpenRouter API key",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            set_key(".env", "OPENAI_API_KEY", api_key)

        all_models = FREE_MODELS + PAID_MODELS
        model = st.selectbox("Model", all_models, index=0)

        language = st.selectbox("Voiceover Language", list(LANGUAGES.keys()), index=0)

        st.divider()
        st.markdown("### Video Settings")
        resolution = st.selectbox("Resolution", ["720p", "1080p"], index=0)
        fps = st.slider("FPS", 12, 30, 24)
        bgm_volume = st.slider("BGM Volume %", 0, 100, 15) / 100.0

        st.divider()
        st.markdown("### History")
        if st.session_state.history:
            for entry in st.session_state.history[:5]:
                ts = entry["timestamp"][:16].replace("T", " ")
                st.caption(f"{entry['title'][:30]}...")
                if os.path.exists(entry["video_path"]):
                    with open(entry["video_path"], "rb") as f:
                        st.download_button(
                            "Re-download", f.read(),
                            file_name=f"{entry['title'][:30]}.mp4",
                            mime="video/mp4",
                            key=f"hist_{entry['timestamp']}",
                        )
        else:
            st.caption("No videos generated yet.")

    st.markdown('<p class="main-title">VideoAgent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-powered professional video generation</p>', unsafe_allow_html=True)

    client = get_client()
    if not client:
        st.error("Please enter your OpenRouter API key in the sidebar to get started.")
        st.stop()

    tab1, tab2 = st.tabs(["Generate Video", "AI Chat"])

    with tab1:
        st.markdown('<div class="step-header">Step 1: Enter Topic</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        topic = col1.text_input("What is your video about?", placeholder="e.g., Artificial Intelligence, Climate Change, Space Exploration...")
        if col2.button("Generate Script", type="primary", use_container_width=True, disabled=not topic):
            with st.spinner("AI is writing your video script..."):
                try:
                    script = generate_script(topic, model, client)
                    st.session_state.script = script
                    st.session_state.step = 2
                    st.success(f"Script generated: {script['title']} ({len(script['segments'])} segments)")
                except Exception as e:
                    st.error(f"Error generating script: {e}")

        if st.session_state.script:
            st.markdown('<div class="step-header">Step 2: Review & Edit Script</div>', unsafe_allow_html=True)
            script = st.session_state.script

            new_title = st.text_input("Video Title", value=script["title"])
            script["title"] = new_title

            seg_data = []
            for i, seg in enumerate(script["segments"]):
                seg_data.append({
                    "#": i + 1,
                    "Narration": seg["narration"],
                    "Image Prompt": seg["image_prompt"],
                })

            edited_df = st.data_editor(
                seg_data,
                num_rows="dynamic",
                use_container_width=True,
                key="script_editor",
                column_config={
                    "#": st.column_config.NumberColumn("#", disabled=True),
                    "Narration": st.column_config.TextColumn("Narration", width="medium"),
                    "Image Prompt": st.column_config.TextColumn("Image Prompt", width="large"),
                },
            )

            updated_segments = []
            for row in updated_df:
                updated_segments.append({
                    "narration": row["Narration"],
                    "image_prompt": row["Image Prompt"],
                })
            script["segments"] = updated_segments
            st.session_state.script = script

            st.markdown('<div class="step-header">Step 3: Generate Video</div>', unsafe_allow_html=True)

            settings = {
                "resolution": resolution,
                "fps": fps,
                "bgm_volume": bgm_volume,
                "language": LANGUAGES[language],
            }

            col_gen, col_status = st.columns([1, 2])
            if col_gen.button("Generate Video", type="primary", use_container_width=True, disabled=st.session_state.generating):
                st.session_state.generating = True
                progress_bar = st.progress(0, text="Starting...")
                status_area = st.empty()

                def update_progress(msg, pct):
                    progress_bar.progress(min(pct, 1.0), text=msg)
                    status_area.markdown(f'<div class="status-box">{msg}</div>', unsafe_allow_html=True)

                try:
                    with st.spinner("This may take several minutes..."):
                        video_path = build_video(script, settings, progress_callback=update_progress)
                    st.session_state.video_path = video_path
                    st.session_state.step = 4
                    progress_bar.progress(1.0, text="Done!")
                    st.session_state.history = save_to_history(script["title"], video_path)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating video: {e}")
                finally:
                    st.session_state.generating = False

            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                st.markdown('<div class="step-header">Step 4: Preview & Download</div>', unsafe_allow_html=True)
                with open(st.session_state.video_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes, format="video/mp4")
                c1, c2 = st.columns(2)
                c1.download_button(
                    "Download MP4", video_bytes,
                    file_name=f"{script['title'][:40]}.mp4",
                    mime="video/mp4", type="primary", use_container_width=True,
                )
                c2.download_button(
                    "Download Script JSON",
                    json.dumps(script, indent=2),
                    file_name="video_script.json",
                    mime="application/json", use_container_width=True,
                )

    with tab2:
        st.markdown('<div class="step-header">AI Chat Refinement</div>', unsafe_allow_html=True)
        st.info("Chat with AI to refine your script. Ask it to shorten, rewrite, add segments, or change the tone.")

        if not st.session_state.script:
            st.warning("Generate a script first in the 'Generate Video' tab.")
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_msg := st.chat_input("Ask AI to modify the script..."):
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)

                with st.chat_message("assistant"):
                    with st.spinner("AI is refining your script..."):
                        try:
                            new_script = refine_script_with_chat(
                                st.session_state.script, user_msg, model, client
                            )
                            st.session_state.script = new_script
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Updated script: **{new_script['title']}** with {len(new_script['segments'])} segments. Go to 'Generate Video' tab to review.",
                            })
                            st.markdown(f"Updated script: **{new_script['title']}** with {len(new_script['segments'])} segments.")
                            st.markdown("Go to **Generate Video** tab to review the changes.")
                        except Exception as e:
                            st.error(f"Error: {e}")

            if st.button("Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()


if __name__ == "__main__":
    main()
