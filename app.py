from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import ollama
import requests
from google.cloud import texttospeech
import googlemaps
import ffmpeg
from moviepy.editor import VideoFileClip, concatenate_videoclips
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.dalle import DalleTools
from agno.utils.common import dataclass_to_dict
from rich.pretty import pprint
from lumaai import LumaAI
import time
from functools import lru_cache
import hashlib
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from location_agent import LocationAgent

# Load environment variables
load_dotenv()

# Retrieve API Keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LUMA_API_KEY = os.getenv("LUMA_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")

print(f"🔑 Google Maps API Key loaded: {'Yes' if GOOGLE_MAPS_API_KEY else 'No'}")
print(f"🔑 API Key length: {len(GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else 0}")
print(f"🔑 API Key first 10 chars: {GOOGLE_MAPS_API_KEY[:10] if GOOGLE_MAPS_API_KEY else 'None'}")

# Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize Google Maps client
try:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    test_result = gmaps.geocode('Paris, France')
    if test_result:
        print("✅ Google Maps client initialized and tested successfully")
        print(f"✅ Test geocoding successful: {test_result[0]['formatted_address']}")
    else:
        print("⚠️ Google Maps client initialized but test geocoding returned no results")
except Exception as e:
    print(f"❌ Error initializing Google Maps client: {str(e)}")
    print(f"❌ Error type: {type(e).__name__}")
    gmaps = None

# Initialize agents
image_agent = Agent(
    model=OpenAIChat(id="gpt-4", api_key=OPENAI_API_KEY),
    tools=[DalleTools()],
    description="AI agent for generating travel images.",
)

location_agent = LocationAgent()

# Serve Frontend
@app.route('/')
def index():
    return render_template('chat.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

@app.route('/itinerary')
def itinerary():
    return render_template('index.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

# Add timing metrics
class TimingMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.model = None
    
    def start(self, model_name):
        self.start_time = time.time()
        self.model = model_name
    
    def end(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

metrics = TimingMetrics()

# Cache size for itineraries (adjust based on your memory constraints)
ITINERARY_CACHE_SIZE = 100

# Create a dictionary to store cached itineraries
itinerary_cache = {}

def get_cached_itinerary(cache_key):
    """Get cached itinerary if it exists."""
    return itinerary_cache.get(cache_key)

def set_cached_itinerary(cache_key, itinerary_text):
    """Store itinerary in cache."""
    if len(itinerary_cache) >= ITINERARY_CACHE_SIZE:
        # Remove oldest item if cache is full
        itinerary_cache.pop(next(iter(itinerary_cache)))
    itinerary_cache[cache_key] = itinerary_text

def generate_cache_key(prompt):
    """Generate a deterministic cache key from the prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()

def optimize_prompt(user_prompt):
    """Optimize the prompt for faster generation."""
    # Extract key information from the prompt
    prompt_parts = user_prompt.split()
    
    # Create a more structured prompt for faster processing
    optimized = (
        f"Create a concise day-by-day itinerary for: {user_prompt}\n"
        "Format: Day X: [Morning/Afternoon/Evening activities]\n"
        "Keep descriptions brief but informative.\n"
        "Include: key attractions, dining, and transportation.\n"
        "Max 3-4 activities per day."
    )
    return optimized

# Generate AI Itinerary using Ollama
@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary():
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        
        if not user_prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Extract destination from prompt
        destination = user_prompt.split(' to ')[1].split(' for')[0].strip() if ' to ' in user_prompt else ''
        os.environ['DESTINATION'] = destination  # Set destination for location agent
        
        print(f"🗺 Generating itinerary for: {user_prompt}")
        
        # Generate itinerary using Ollama
        response = ollama.chat(
            model="mistral",
            messages=[{
                "role": "user", 
                "content": f"Create a detailed 3-day itinerary for {user_prompt}. Include specific attractions, landmarks, and activities. Format each day with 'Day X:' and list 3-4 activities per day. Make sure to mention specific places to visit."
            }],
            options={
                "temperature": 0.7,
                "num_predict": 512
            }
        )
        
        itinerary_text = response.get("message", {}).get("content", "No itinerary generated.")
        
        return jsonify({
            'itinerary': itinerary_text,
            'destination': destination
        })
    except Exception as e:
        print(f"❌ Error generating itinerary: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

def extract_locations_from_itinerary(itinerary_text: str, base_location: str) -> List[Dict]:
    """Extract location names from itinerary text and get their coordinates."""
    # Use the AI model to extract location names
    prompt = f"""
    Extract all specific location names (attractions, landmarks, restaurants, etc.) from this itinerary.
    Base location: {base_location}
    Itinerary:
    {itinerary_text}
    
    Return only a JSON array of location names, like:
    ["Location 1", "Location 2", "Location 3"]
    """
    
    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        
        # Extract JSON array from response
        content = response.get("message", {}).get("content", "[]")
        # Find JSON array in the response
        json_match = re.search(r'\[.*\]', content)
        if json_match:
            locations = json.loads(json_match.group())
        else:
            locations = []
        
        # Get coordinates for each location
        location_data = []
        
        for location in locations:
            try:
                # Add base location to search query for better accuracy
                search_query = f"{location}, {base_location}"
                result = gmaps.geocode(search_query)
                
                if result:
                    location_data.append({
                        "name": location,
                        "lat": result[0]['geometry']['location']['lat'],
                        "lng": result[0]['geometry']['location']['lng'],
                        "description": f"Visit {location} during your trip to {base_location}",
                        # We'll get the image URL from the image generation later
                        "image": None
                    })
            except Exception as e:
                print(f"Error geocoding {location}: {str(e)}")
                continue
        
        return location_data
    
    except Exception as e:
        print(f"Error extracting locations: {str(e)}")
        return []

@app.route('/extract_locations', methods=['POST'])
def extract_locations():
    try:
        data = request.get_json()
        itinerary = data.get('itinerary', '')
        
        # Use the location agent to process the itinerary
        locations = location_agent.process_itinerary(itinerary)
        
        return jsonify({
            'locations': locations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask App
if __name__ == '__main__':
    # Test Google Maps client
    try:
        test_result = gmaps.geocode('Paris, France')
        if test_result:
            print("✅ Google Maps client initialized and tested successfully")
            print("✅ Test geocoding successful: Paris, France")
    except Exception as e:
        print(f"❌ Google Maps client test failed: {str(e)}")
        raise

    app.run(debug=True)

