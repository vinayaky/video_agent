import os, json, base64, wave, struct, math
from dotenv import load_dotenv
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
client = OpenAI()

WIDTH, HEIGHT = 1280, 720
FPS = 24
FADE_DURATION = 0.5
FONT_PATH = "C:/Windows/Fonts/arial.ttf"
FONT_PATH_BOLD = "C:/Windows/Fonts/arialbd.ttf"

os.makedirs("audio", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)

with open("video_script.json", "r", encoding="utf-8") as f:
    script = json.load(f)

segments = script["segments"]
title_text = script["title"]


def generate_ai_image(prompt, filename):
    print(f"  Generating AI image: {prompt[:60]}...")
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
                img_bytes = base64.b64decode(b64_data)
                with open(filename, "wb") as f:
                    f.write(img_bytes)
                print(f"  AI image saved: {filename}")
                return True
    except Exception:
        pass

    print("  Using styled gradient image")
    return False


def make_fallback_image(filename, prompt=""):
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
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    font = ImageFont.truetype(FONT_PATH_BOLD, 48)
    words = prompt.split()[:8]
    short = " ".join(words) + ("..." if len(prompt.split()) > 8 else "")
    bbox = draw.textbbox((0, 0), short, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 30), short, fill=(255, 255, 255, 180), font=font)
    img.save(filename)


def make_subtitle_image(text, size=(WIDTH, 140)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), size], fill=(0, 0, 0, 160))

    font = ImageFont.truetype(FONT_PATH, 28)
    margin = 40
    max_w = size[0] - 2 * margin
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
    y = (size[1] - total_h) // 2
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        x = (size[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), ln, fill=(255, 255, 255, 255), font=font)
        y += line_height

    return np.array(img)


def make_title_clip(title_str, duration=4):
    bg = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 30))
    draw = ImageDraw.Draw(bg)
    font_big = ImageFont.truetype(FONT_PATH_BOLD, 52)
    font_small = ImageFont.truetype(FONT_PATH, 22)

    bbox = draw.textbbox((0, 0), title_str, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 60), title_str, fill="white", font=font_big)

    line_y = HEIGHT // 2 + 40
    draw.line([(WIDTH // 2 - 100, line_y), (WIDTH // 2 + 100, line_y)], fill=(233, 69, 96), width=3)

    sub = "AI Generated Video"
    bbox2 = draw.textbbox((0, 0), sub, font=font_small)
    sw = bbox2[2] - bbox2[0]
    draw.text(((WIDTH - sw) // 2, line_y + 20), sub, fill=(180, 180, 180), font=font_small)

    clip = ImageClip(np.array(bg)).with_duration(duration)
    clip = clip.with_effects([vfx.FadeIn(1.0), vfx.FadeOut(1.0)])
    return clip


def make_end_clip(duration=3):
    bg = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 30))
    draw = ImageDraw.Draw(bg)
    font = ImageFont.truetype(FONT_PATH_BOLD, 44)
    text = "Thank You For Watching"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 40), text, fill="white", font=font)

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


print(f"=== Generating Professional Video: {title_text} ===\n")

clips = []

for i, seg in enumerate(segments):
    print(f"[{i + 1}/{len(segments)}] {seg['narration'][:50]}...")

    img_file = f"images/segment{i + 1}.jpg"
    if not generate_ai_image(seg["image_prompt"], img_file):
        make_fallback_image(img_file, seg["image_prompt"])

    audio_file = f"audio/voiceover{i + 1}.mp3"
    tts = gTTS(text=seg["narration"], lang="en", slow=False)
    tts.save(audio_file)

    audio_clip = AudioFileClip(audio_file)
    audio_clip = audio_clip.with_effects([afx.AudioFadeIn(0.3), afx.AudioFadeOut(0.3)])
    dur = audio_clip.duration

    img = Image.open(img_file).resize((WIDTH, HEIGHT), Image.LANCZOS)
    img_arr = np.array(img)

    img_clip = ImageClip(img_arr).with_duration(dur)

    sub_img = make_subtitle_image(seg["narration"])
    sub_clip = ImageClip(sub_img).with_duration(dur).with_position(("center", HEIGHT - 140))

    composite = CompositeVideoClip([img_clip, sub_clip], size=(WIDTH, HEIGHT)).with_duration(dur)
    composite = composite.with_audio(audio_clip)
    composite = composite.with_effects([vfx.FadeIn(FADE_DURATION), vfx.FadeOut(FADE_DURATION)])

    clips.append(composite)
    print(f"  Clip {i + 1} done (duration: {dur:.1f}s)")

print("\nBuilding title screen...")
title_clip = make_title_clip(title_text)
clips.insert(0, title_clip)

print("Building end screen...")
end_clip = make_end_clip()
clips.append(end_clip)

print("\nConcatenating clips with crossfade...")
final = concatenate_videoclips(clips, method="compose", padding=-FADE_DURATION)

print("Generating background music...")
bgm_file = "audio/bgm.wav"
generate_bgm(final.duration + 2, bgm_file)
bgm = AudioFileClip(bgm_file).with_duration(final.duration)
bgm = bgm.with_effects([afx.AudioFadeIn(1.0), afx.AudioFadeOut(1.0)])
bgm_low = bgm.with_volume_scaled(0.15)

voiceover = final.audio
mixed_audio = CompositeAudioClip([voiceover, bgm_low])
final = final.with_audio(mixed_audio)

print("Rendering final video (this may take a while)...")
final.write_videofile(
    "final_video.mp4",
    fps=FPS,
    codec="libx264",
    audio_codec="aac",
    preset="ultrafast",
    bitrate="3000k",
)

print("\nDone! Professional video saved as final_video.mp4")
