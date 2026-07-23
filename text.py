import os, json
from openai import OpenAI

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY environment variable first")
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")

client = OpenAI()

topic = input("What topic do you want the video to be about: ")

prompt = f"""Create a short video script about: {topic}

Return ONLY valid JSON with these fields:
- title: A catchy video title
- segments: An array of objects, each with:
  - narration: The voiceover text (1-2 sentences, conversational tone)
  - image_prompt: A detailed description for AI image generation (visual scene, no text in image)

Create 5-7 segments. Make the narration engaging and informative.
The image prompts should describe visually rich scenes that match each narration point.
Return ONLY the JSON, no markdown fences, no explanation."""

print("Generating video script...")
completions = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=2048,
    temperature=0.7,
)

response_text = completions.choices[0].message.content.strip()

if response_text.startswith("```"):
    response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

script = json.loads(response_text)

with open("generated_text.txt", "w", encoding="utf-8") as f:
    f.write(script["title"] + "\n\n")
    for seg in script["segments"]:
        f.write(seg["narration"] + "\n\n")

with open("video_script.json", "w", encoding="utf-8") as f:
    json.dump(script, f, indent=2, ensure_ascii=False)

print(f"Title: {script['title']}")
print(f"Generated {len(script['segments'])} segments")
print("Files saved: generated_text.txt, video_script.json")
