from flask import Flask, render_template, request, jsonify, send_file
import os
import ollama
import requests
from google.cloud import texttospeech
from load_config import load_config
import ffmpeg
from moviepy import VideoFileClip, concatenate_videoclips
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.dalle import DalleTools
from agno.utils.common import dataclass_to_dict
from rich.pretty import pprint
from lumaai import LumaAI
import time

def load_config():
    config_path = "config.txt"  # Or .env, secrets.env, etc.

    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        return

    with open(config_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or "=" not in line:  # Ignore empty/malformed lines
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value  # Set as environment variable

# Load config when script runs
load_config()

#Retrieve API Keys
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_your_tts.json"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LUMA_API_KEY = os.getenv("LUMA_API_KEY")

# ✅ Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")


# ------------------------------------------------------------------
# 4) Existing "image_agent" for generating images (agno)
# ------------------------------------------------------------------
image_agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[DalleTools()],
    description="AI agent for generating travel images.",
)

# Serve Frontend
@app.route('/')
def index():
    return render_template('index.html')


# Generate AI Itinerary using Ollama
@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary():
    data = request.json
    user_prompt = data.get("prompt")

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    print(f"🗺 Generating itinerary for: {user_prompt}")

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": f"Create a detailed itinerary for {user_prompt}"}]
        )
        itinerary_text = response.get("message", {}).get("content", "No itinerary generated.")

        print("Generated Itinerary:", itinerary_text)  # Debugging output

        return jsonify({"itinerary": itinerary_text})

    except Exception as e:
        print(f"❌ Ollama API Error: {str(e)}")
        return jsonify({"error": "Failed to generate itinerary"}), 500

@app.route('/generate_images', methods=['POST'])
def generate_images():
    data = request.json
    location = data.get("location", "default destination")

    # ✅ Run Agent to Generate Images
    run_stream = image_agent.run(
        f"Create 6 different images of the travel destination, {location}",
        stream=True,
        stream_intermediate_steps=True,
    )

    image_urls = []

    for chunk in run_stream:
        print(f"🔄 Processing chunk: {chunk}")  # Debugging Output

        # ✅ Extract from 'images' if available
        if hasattr(chunk, "images") and chunk.images:
            for img in chunk.images:
                if hasattr(img, "url"):
                    image_urls.append(img.url)

        # ✅ Extract from 'tools' if available
        if hasattr(chunk, "tools") and chunk.tools:
            for tool in chunk.tools:
                if "content" in tool and "Image has been generated at the URL" in tool["content"]:
                    # Extract the actual URL from the response text
                    image_url = tool["content"].split("URL ")[1].strip()
                    image_urls.append(image_url)

    # ✅ Ensure at least one image was found
    if not image_urls:
        return jsonify({"error": "Failed to generate images"}), 500

    return jsonify({"image_urls": image_urls})



@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        data = request.json
        user_prompt = data.get("prompt", "a 3-day trip exploring scenic landscapes and iconic landmarks")

        # 🔥 Compose a detailed cinematic prompt
        prompt = (
            f"A smooth cinematic travel video showing multiple iconic landmarks and scenic views during a {user_prompt}. "
            "Include famous attractions, local culture, and a beautiful sunset. Multiple camera angles, lively streets, "
            "and relaxing vibes."
        )

        # Pass your auth token directly 
        client = LumaAI(auth_token=os.getenv("LUMA_API_KEY"))

        generation = client.generations.create(prompt=prompt)

        for _ in range(20):  # Poll up to 1 minute (20 x 3s)
            generation = client.generations.get(id=generation.id)
            if generation.state == "completed":
                video_url = generation.assets.video
                print(f"✅ Video ready: {video_url}")
                return jsonify({"video_url": video_url})
            elif generation.state == "failed":
                print(f"❌ Generation failed: {generation.failure_reason}")
                return jsonify({"error": f"Generation failed: {generation.failure_reason}"}), 500
            time.sleep(3)

        return jsonify({"error": "Video generation timed out."}), 500

    except Exception as e:
        print(f"❌ Luma API Exception: {e}")
        return jsonify({"error": "Failed to generate video from Luma AI."}), 500



# Generate AI Voiceover for the Itinerary
@app.route('/generate_voiceover', methods=['POST'])
def generate_voiceover():
    data = request.json
    itinerary_text = data.get("text")  # ✅ Get the full itinerary text

    if not itinerary_text:
        return jsonify({"error": "No text provided"}), 400

    print(f"🔊 Generating voiceover for itinerary text: {itinerary_text}")

    try:
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=itinerary_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

        audio_file = "static/ai_voice.mp3"
        with open(audio_file, "wb") as out:
            out.write(response.audio_content)

        print(f"✅ Voiceover saved at: {audio_file}")

        # Return direct path to the file (avoiding Flask's `/play_voiceover`)
        return jsonify({"audio_url": f"/static/ai_voice.mp3"})

    except Exception as e:
        print(f"❌ Google TTS Error: {str(e)}")
        return jsonify({"error": "Failed to generate voiceover"}), 500



# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)

