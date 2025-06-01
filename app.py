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
from agno.models.message import Message
from rich.pretty import pprint
from lumaai import LumaAI
import time
from functools import lru_cache
import hashlib
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime, timedelta # Added for date calculations

# Load environment variables
load_dotenv()

# Retrieve API Keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LUMA_API_KEY = os.getenv("LUMA_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
# GOOGLE_CUSTOM_SEARCH_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY') # Removed
# PROGRAMMABLE_SEARCH_ENGINE_ID = os.getenv('PROGRAMMABLE_SEARCH_ENGINE_ID') # Removed

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")
# if not GOOGLE_CUSTOM_SEARCH_API_KEY: # Removed
# print("Warning: GOOGLE_CUSTOM_SEARCH_API_KEY not found. Map pin images from Google Search will not work.") # Removed
# if not PROGRAMMABLE_SEARCH_ENGINE_ID: # Removed
# print("Warning: PROGRAMMABLE_SEARCH_ENGINE_ID not found. Map pin images from Google Search will not work.") # Removed

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

# Initialize CSS generation model
css_model = OpenAIChat(id="gpt-3.5-turbo", api_key=OPENAI_API_KEY) # Using gpt-3.5-turbo for CSS

# Serve Frontend
@app.route('/')
def index():
    return render_template('chat.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

@app.route('/itinerary')
def itinerary():
    default_image_url = "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=1200&q=80"
    default_body_css = "background-color: #87CEEB;" # Sky blue
    return render_template('index.html', 
                           google_maps_api_key=GOOGLE_MAPS_API_KEY, 
                           default_image_url=default_image_url,
                           default_body_css=default_body_css)

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

def is_valid_destination(destination):
    generic = {"yes", "no", "maybe", "idk", "not sure", "help", "okay", "ok"}
    if not destination or len(destination) < 3:
        return False
    if destination.lower() in generic:
        return False
    if destination.isdigit():
        return False
    return True

# Generate AI Itinerary using Ollama
@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary():
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        destination = user_prompt.strip()
        if not is_valid_destination(destination):
            return jsonify({"error": "Please enter a valid destination (e.g., 'Paris', 'Tokyo', 'New York')."}), 400
        # Validate that the destination is a real destination (using geocoding)
        geocode_result = gmaps.geocode(destination)
        if not geocode_result:
             return jsonify({"error": "\"" + destination + "\" is not a real destination. Please enter a valid destination (e.g., 'Paris', 'Tokyo', 'New York')."}), 400
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
    print("\n[DEBUG] /generate_images: Route called.")
    try:
        data = request.json
        location = data.get("location", "default destination")
        print(f"[DEBUG] /generate_images: Location received: '{location}'")

        image_generation_prompt = f"Create 6 different images of the travel destination, {location}"
        print(f"[DEBUG] /generate_images: Prompt for image_agent: '{image_generation_prompt}'")

        # ✅ Run Agent to Generate Images
        run_stream = image_agent.run(
            image_generation_prompt,
            stream=True,
            stream_intermediate_steps=True,
        )

        image_urls = []
        print("[DEBUG] /generate_images: Initialized empty image_urls list.")

        for chunk in run_stream:
            print(f"[DEBUG] /generate_images: Processing chunk: {chunk}")

            # ✅ Extract from 'images' if available
            if hasattr(chunk, "images") and chunk.images:
                for img_idx, img in enumerate(chunk.images):
                    if hasattr(img, "url"):
                        image_urls.append(img.url)
                        print(f"[DEBUG] /generate_images: Extracted URL from chunk.images[{img_idx}]: {img.url}")

            # ✅ Extract from 'tools' if available
            if hasattr(chunk, "tools") and chunk.tools:
                for tool_idx, tool in enumerate(chunk.tools):
                    if "content" in tool and "Image has been generated at the URL" in tool["content"]:
                        # Extract the actual URL from the response text
                        try:
                            image_url = tool["content"].split("URL ")[1].strip()
                            image_urls.append(image_url)
                            print(f"[DEBUG] /generate_images: Extracted URL from chunk.tools[{tool_idx}]: {image_url}")
                        except IndexError:
                            print(f"[WARN] /generate_images: Could not parse URL from tool content: {tool['content']}")


        if not image_urls:
            print("[DEBUG] /generate_images: No image URLs were extracted. Returning error.")
            return jsonify({"error": "Failed to generate images, no URLs found after processing."}), 500

        print(f"[DEBUG] /generate_images: Successfully extracted image URLs: {image_urls}")
        return jsonify({"image_urls": image_urls})

    except Exception as e:
        import traceback
        print(f"[ERROR] /generate_images: Unhandled exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred in generate_images: {str(e)}"}), 500

@app.route('/generate_video', methods=['POST'])
def generate_video():
    print("\n[DEBUG] /generate_video: Route called.")
    try:
        data = request.json
        user_prompt = data.get("prompt", "a 3-day trip exploring scenic landscapes and iconic landmarks")
        print(f"[DEBUG] /generate_video: User prompt received: '{user_prompt}'")

        prompt = (
            f"A smooth cinematic travel video showing multiple iconic landmarks and scenic views during a {user_prompt}. "
            "Include famous attractions, local culture, and a beautiful sunset. Multiple camera angles, lively streets, "
            "and relaxing vibes."
        )
        print(f"[DEBUG] /generate_video: Full prompt for LumaAI: '{prompt}'")

        if not LUMA_API_KEY:
            print("[ERROR] /generate_video: LUMA_API_KEY is not set. Cannot generate video.")
            return jsonify({"error": "Luma API key not configured."}), 500

        client = LumaAI(auth_token=LUMA_API_KEY)
        print("[DEBUG] /generate_video: LumaAI client initialized.")

        print("[DEBUG] /generate_video: Sending request to LumaAI to create generation...")
        generation = client.generations.create(
            prompt=prompt,
            model="ray-2",
            resolution="720p",
            duration="5s",
            aspect_ratio="16:9"
        )
        print(f"[DEBUG] /generate_video: LumaAI generation object created: ID {generation.id}, State: {generation.state}")

        # Polling loop
        for i in range(40): # Max 40 attempts (2 minutes with 3s sleep)
            print(f"[DEBUG] /generate_video: Polling attempt {i+1}/40 for generation ID {generation.id}...")
            time.sleep(3) # Sleep before getting status
            generation = client.generations.get(id=generation.id)
            print(f"[DEBUG] /generate_video: 🔄 Generation state: {generation.state}")
            
            if generation.state == "completed":
                video_url = generation.assets.video
                print(f"[DEBUG] /generate_video: ✅ Video generation completed. URL: {video_url}")
                return jsonify({"video_url": video_url})
            elif generation.state == "failed":
                failure_reason = generation.failure_reason or "Unknown reason"
                print(f"[ERROR] /generate_video: ❌ Generation failed: {failure_reason}")
                return jsonify({"error": f"LumaAI generation failed: {failure_reason}"}), 500
            elif generation.state in ["processing", "pending"]:
                print(f"[DEBUG] /generate_video: Video still processing (state: {generation.state}). Continuing to poll.")
            else:
                print(f"[WARN] /generate_video: Unknown LumaAI generation state: {generation.state}")


        print(f"[WARN] /generate_video: ⏱️ Video generation timed out after {40*3} seconds for ID {generation.id}.")
        return jsonify({"error": "Video generation timed out."}), 500

    except Exception as e:
        import traceback
        print(f"[ERROR] /generate_video: ❌ Luma API Exception or other error: {repr(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to generate video from Luma AI: {str(e)}"}), 500

# Generate AI Voiceover for the Itinerary
@app.route('/generate_voiceover', methods=['POST'])
def generate_voiceover():
    print("[DEBUG] /generate_voiceover: Route called.")
    try:
        data = request.json
        if not data:
            print("[DEBUG] /generate_voiceover: No JSON data received.")
            return jsonify({"error": "No JSON data provided"}), 400
        
        itinerary_text = data.get("text")
        if not itinerary_text:
            print("[DEBUG] /generate_voiceover: 'text' field missing from JSON data.")
            return jsonify({"error": "No text provided in JSON"}), 400

        print(f"[DEBUG] /generate_voiceover: Attempting to generate voiceover for text (first 100 chars): {itinerary_text[:100]}")

        # Ensure Google Cloud credentials are set up correctly if not done globally
        # For example, os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"
        # This should ideally be handled at application startup or via environment.

        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=itinerary_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            # Consider trying a specific voice name if NEUTRAL causes issues:
            # name="en-US-Wavenet-D" 
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            # You can adjust speaking rate and pitch if desired
            # speaking_rate=1.0,
            # pitch=0
        )

        print("[DEBUG] /generate_voiceover: Synthesizing speech...")
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

        audio_file = "static/ai_voice.mp3"
        print(f"[DEBUG] /generate_voiceover: Writing audio content to {audio_file}")
        with open(audio_file, "wb") as out:
            out.write(response.audio_content)

        print(f"[DEBUG] /generate_voiceover: Voiceover saved successfully.")
        return jsonify({"audio_url": f"/{audio_file}"}) # Ensure leading slash for correct URL path

    except Exception as e:
        import traceback
        print(f"[DEBUG] /generate_voiceover: !!! ERROR IN ROUTE !!!")
        print(traceback.format_exc()) # Print full traceback
        # Return a more specific error message if possible
        error_message = f"Failed to generate voiceover: {str(e)}"
        if "GOOGLE_APPLICATION_CREDENTIALS" in str(e):
            error_message = "Failed to generate voiceover: Google Cloud authentication error. Ensure credentials are set up."
        
        return jsonify({"error": error_message}), 500

def extract_locations_from_itinerary(itinerary_text: str, base_location: str) -> List[Dict]:
    """Extract location names, get coordinates, and find an image URL via Google Places API."""
    print(f"[DEBUG] extract_locations_from_itinerary: Called with base_location: '{base_location}'")
    print(f"[DEBUG] extract_locations_from_itinerary: Itinerary snippet: {itinerary_text[:200]}...")
    
    prompt = f"""
    Extract all specific location names (attractions, landmarks, restaurants, etc.) from this itinerary.
    Base location: {base_location}
    Itinerary:
    {itinerary_text}
    
    Return only a JSON array of location names, like:
    ["Location 1", "Location 2", "Location 3"]
    """
    print("[DEBUG] extract_locations_from_itinerary: Sending prompt to Ollama for location name extraction.")
    
    extracted_names = []
    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        content = response.get("message", {}).get("content", "[]")
        print(f"[DEBUG] extract_locations_from_itinerary: Ollama response content for names: {content}")
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                extracted_names = json.loads(json_match.group())
                print(f"[DEBUG] extract_locations_from_itinerary: Extracted location names: {extracted_names}")
            except json.JSONDecodeError as json_e:
                print(f"[DEBUG] extract_locations_from_itinerary: JSONDecodeError parsing names: {json_e}. Content: {json_match.group()}")
        else:
            print("[DEBUG] extract_locations_from_itinerary: No JSON array in Ollama response for names.")
    except Exception as e:
        print(f"[DEBUG] extract_locations_from_itinerary: Error during Ollama call for names: {str(e)}")

    location_data = []
    if not gmaps:
        print("[DEBUG] extract_locations_from_itinerary: Google Maps client (gmaps) not initialized. Skipping geocoding & image search.")
        return []
    
    # Check if necessary keys for Google Image Search are present - Removed
    # can_search_images = GOOGLE_CUSTOM_SEARCH_API_KEY and PROGRAMMABLE_SEARCH_ENGINE_ID # Removed
    # if not can_search_images: # Removed
    # print("[DEBUG] extract_locations_from_itinerary: API key or CX for Google Custom Search not set. Skipping image search for pins.") # Removed

    for name in extracted_names:
        if not isinstance(name, str) or not name.strip():
            print(f"[DEBUG] extract_locations_from_itinerary: Skipping invalid location name: {name}")
            continue
        
        location_name = name.strip()
        lat, lng, description = None, None, None
        image_url = None # Default to no image

        try:
            search_query_maps = f"{location_name}, {base_location}"
            print(f"[DEBUG] extract_locations_from_itinerary: Geocoding '{search_query_maps}'")
            geocode_result = gmaps.geocode(search_query_maps)
            
            if geocode_result and len(geocode_result) > 0:
                res = geocode_result[0]
                lat = res['geometry']['location']['lat']
                lng = res['geometry']['location']['lng']
                description = f"Visit {location_name} in {base_location}."
                place_id = res.get('place_id') # Get place_id
                print(f"[DEBUG] extract_locations_from_itinerary: Geocoded '{location_name}' to ({lat}, {lng}), Place ID: {place_id}")

                if place_id and GOOGLE_MAPS_API_KEY:
                    print(f"[DEBUG] extract_locations_from_itinerary: Attempting to fetch photo for Place ID: {place_id}")
                    try:
                        # Fetch place details to get photo reference
                        place_details = gmaps.place(place_id=place_id, fields=['name', 'photo'])
                        photos_data = place_details.get('result', {}).get('photos')
                        
                        if photos_data and len(photos_data) > 0:
                            photo_reference = photos_data[0].get('photo_reference')
                            if photo_reference:
                                # Construct the photo URL
                                # You can adjust maxwidth or maxheight as needed
                                image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_MAPS_API_KEY}"
                                print(f"[DEBUG] extract_locations_from_itinerary: Constructed image URL for '{location_name}': {image_url}")
                            else:
                                print(f"[DEBUG] extract_locations_from_itinerary: No photo_reference found for Place ID {place_id} for '{location_name}'.")
                        else:
                            print(f"[DEBUG] extract_locations_from_itinerary: No photos array found for Place ID {place_id} for '{location_name}'.")
                    except Exception as e_place_photo:
                        print(f"[DEBUG] extract_locations_from_itinerary: Error fetching Google Place photo for '{location_name}': {e_place_photo}")
                else:
                    if not place_id:
                        print(f"[DEBUG] extract_locations_from_itinerary: No Place ID found for '{location_name}', cannot fetch Google Place photo.")
                    if not GOOGLE_MAPS_API_KEY:
                         print(f"[DEBUG] extract_locations_from_itinerary: GOOGLE_MAPS_API_KEY not set, cannot fetch Google Place photo for '{location_name}'.")
            
                location_data.append({
                    "name": location_name,
                    "lat": lat,
                    "lng": lng,
                    "description": description,
                    "image": image_url # This will be None if no image was found
                })
            else:
                print(f"[DEBUG] extract_locations_from_itinerary: Geocoding failed for '{location_name}'. No results.")
        
        except Exception as e:
            print(f"[DEBUG] extract_locations_from_itinerary: Error processing location '{name}': {str(e)}")
            continue
        
    print(f"[DEBUG] extract_locations_from_itinerary: Returning {len(location_data)} locations.")
    return location_data

@app.route('/extract_locations', methods=['POST'])
def extract_locations():
    print("\n🔥🔥 HELLO WORLD FROM /EXTRACT_LOCATIONS! ROUTE HIT 🔥🔥\n") # Prominent print statement
    try:
        print("[DEBUG] /extract_locations: Route called.")
        data = request.get_json()
        if not data:
            print("[DEBUG] /extract_locations: No JSON data received in request.")
            return jsonify({'error': 'Invalid request: No JSON data.'}), 400
            
        itinerary = data.get('itinerary', '')
        base_location = data.get('location', '')
        print(f"[DEBUG] /extract_locations: Received itinerary (snippet): {itinerary[:100] if itinerary else 'None'}... Base location: '{base_location}'")

        if not itinerary:
            print("[DEBUG] /extract_locations: Itinerary text is missing.")
            return jsonify({'error': 'Itinerary text is required.'}), 400
        if not base_location:
            print("[DEBUG] /extract_locations: Base location is missing. Attempting fallback.")
            base_location = os.getenv('DESTINATION', 'the planned destination') # Fallback
            print(f"[DEBUG] /extract_locations: Fallback base location: '{base_location}'")

        print(f"[DEBUG] /extract_locations: Calling extract_locations_from_itinerary.")
        locations = extract_locations_from_itinerary(itinerary_text=itinerary, base_location=base_location)
        
        print(f"[DEBUG] /extract_locations: extract_locations_from_itinerary returned {len(locations)} locations.")
        return jsonify({
            'locations': locations
        })
    except Exception as e:
        # Log the full exception traceback for detailed debugging
        import traceback
        print(f"[DEBUG] /extract_locations: !!! UNHANDLED ERROR IN ROUTE !!!")
        print(traceback.format_exc()) # This will print the full stack trace
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/randomize_background_css', methods=['POST'])
def randomize_background_css():
    try:
        data = request.get_json()
        theme_prompt = data.get('theme_prompt', 'default nature theme')
        print(f'[DEBUG] /randomize_background_css: Received theme prompt: "{theme_prompt}"')

        final_image_url = None
        body_css_rules = "background-color: #f0f0f0;" # Default fallback

        # --- Get image URL from Google Places API ---
        if gmaps and GOOGLE_MAPS_API_KEY:
            try:
                print(f"[DEBUG] /randomize_background_css: Searching Google Places for theme: '{theme_prompt}'")
                # Use find_place to get the most relevant place for the theme
                # Request fields: place_id (to confirm a place) and photos (to get photo references)
                places_result = gmaps.find_place(input=theme_prompt, 
                                                 input_type='textquery', 
                                                 fields=['place_id', 'photos', 'name'])

                if places_result and places_result.get('candidates'):
                    top_candidate = places_result['candidates'][0]
                    place_name = top_candidate.get('name', 'the selected place')
                    print(f"[DEBUG] /randomize_background_css: Found place candidate: {place_name}")
                    
                    if 'photos' in top_candidate and top_candidate['photos']:
                        photo_reference = top_candidate['photos'][0].get('photo_reference')
                        if photo_reference:
                            final_image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=1920&photoreference={photo_reference}&key={GOOGLE_MAPS_API_KEY}"
                            print(f'[DEBUG] /randomize_background_css: Constructed Google Places Photo URL: "{final_image_url}"')
                        else:
                            print(f"[DEBUG] /randomize_background_css: Place '{place_name}' found, but no photo_reference available in the first photo object.")
                    else:
                        print(f"[DEBUG] /randomize_background_css: Place '{place_name}' found, but no 'photos' array or it's empty.")
                else:
                    print(f"[DEBUG] /randomize_background_css: No place candidates found by Google Places for theme '{theme_prompt}'.")
            except Exception as e_gmaps:
                print(f"[ERROR] /randomize_background_css: Error during Google Places API call: {e_gmaps}")
                final_image_url = None # Ensure it's None if Places API call fails
        else:
            print("[WARN] /randomize_background_css: Google Maps client (gmaps) or API key not available. Cannot fetch image from Google Places.")

        # --- Get base CSS properties from AI (still used for fallback color) ---
        system_message_content = f'''You are a UI styling assistant.
Your task is to provide fallback CSS `base_css_properties` for a webpage background, given a theme.
The theme is: '{theme_prompt}'.

Respond ONLY with a valid JSON object containing the 'base_css_properties'.
Do NOT include an 'image_url' in your response.

Example for theme "ocean":
{{
  "base_css_properties": "background-color: #0077be;"
}}
'''
        system_message = Message(role="system", content=system_message_content)
        user_message = Message(role="user", content=f"Generate base_css_properties for the theme: {theme_prompt}.")

        print(f'[DEBUG] /randomize_background_css: Sending request to AI for CSS properties with theme: "{theme_prompt}"')
        
        raw_model_content_css = ""
        try:
            response_data_css = css_model.invoke(messages=[system_message, user_message])
            if (
                response_data_css and 
                hasattr(response_data_css, 'choices') and 
                response_data_css.choices and 
                len(response_data_css.choices) > 0 and 
                hasattr(response_data_css.choices[0], 'message') and 
                response_data_css.choices[0].message and 
                hasattr(response_data_css.choices[0].message, 'content') and 
                isinstance(response_data_css.choices[0].message.content, str)
            ):
                raw_model_content_css = response_data_css.choices[0].message.content
                print(f'[DEBUG] /randomize_background_css: Raw model content for CSS: "{raw_model_content_css}"')
                parsed_css_response = json.loads(raw_model_content_css)
                body_css_rules = parsed_css_response.get('base_css_properties', "background-color: #FFFFFF;")
                if not body_css_rules.strip().endswith(';'):
                    body_css_rules += ';'
            else:
                print("[ERROR] /randomize_background_css: Failed to generate CSS from AI in expected format.")
        
        except json.JSONDecodeError:
            print(f"[ERROR] /randomize_background_css: Failed to parse JSON for CSS from AI: {raw_model_content_css}")
            if "background" in raw_model_content_css.lower() and "{" not in raw_model_content_css and "<" not in raw_model_content_css:
                body_css_rules = raw_model_content_css.strip()
                if not body_css_rules.endswith(';'): body_css_rules += ';'
        except Exception as e_css_ai:
            print(f"[ERROR] /randomize_background_css: Error processing AI response for CSS: {e_css_ai}")

        print(f'[DEBUG] /randomize_background_css: Body CSS rules to send to client: "{body_css_rules}"')
        print(f'[DEBUG] /randomize_background_css: Image URL to send to client (from Google Places): "{final_image_url}"')
        
        try:
            with open(os.path.join(app.static_folder, 'generated_background_body.css'), 'w') as f:
                f.write(f"/* Body CSS properties from AI: */\\nbody {{ {body_css_rules} }}\\n/* Image URL from Google Places: {final_image_url if final_image_url else 'None'} */")
            print(f'[DEBUG] /randomize_background_css: Saved body CSS and image URL info to "{os.path.join(app.static_folder, "generated_background_body.css")}"')
        except Exception as e:
            print(f"[ERROR] /randomize_background_css: Could not write to generated_background_body.css: {e}")

        return jsonify({'body_css_rules': body_css_rules, 'image_url': final_image_url})

    except Exception as e:
        print(f"[ERROR] /randomize_background_css: Unexpected error in route: {e}")
        import traceback
        traceback.print_exc()
        # Ensure a valid JSON response even in case of unexpected errors before jsonify
        return jsonify({'body_css_rules': 'background-color: #f0f0f0;', 'image_url': None, 'error': f'Internal server error: {str(e)}'}), 500

@app.route('/generate_budget_widget', methods=['POST'])
def generate_budget_widget():
    try:
        data = request.get_json()
        destination = data.get('destination')
        travel_style = data.get('travel_style', 'balanced budget') # Default to balanced

        if not destination:
            return jsonify({'error': 'Destination is required for budget widget.'}), 400

        # Calculate dates for next week
        today = datetime.today()
        start_of_next_week = today + timedelta(days=(7 - today.weekday())) # Next Monday
        end_of_next_week = start_of_next_week + timedelta(days=6) # Next Sunday
        date_range_str = f"{start_of_next_week.strftime('%b %d')} - {end_of_next_week.strftime('%b %d, %Y')}"

        print(f"[DEBUG] /generate_budget_widget: Dest: {destination}, Style: {travel_style}, Dates: {date_range_str}")

        widget_html = "<!-- Widget could not be generated -->"
        prompt_content = ""

        if travel_style.lower() == 'balanced budget':
            prompt_content = f"""
            Create HTML for a flight ticket tracker widget for a trip to {destination} for next week ({date_range_str}).
            The widget should display 3-4 FAKE economy class flight options.
            For each option, include: Airline (e.g., 'SkyHigh Airways', 'BudgetJet'), Stops (e.g., '1 stop', 'Non-stop'), and Price (e.g., '$350 - $450').
            Use simple, clean HTML. Make it look like a compact tracker or list. Example structure for one item:
            <div>
                <h4>SkyHigh Airways</h4>
                <p>Dates: {date_range_str}</p>
                <p>Stops: 1 stop</p>
                <p>Price: $420</p>
            </div>
            Wrap all options in a single container div with a class 'budget-flight-tracker'.
            Do NOT include any CSS or <style> tags. Only provide the raw HTML for the content of the widget.
            """
        elif travel_style.lower() == 'luxury travel':
            prompt_content = f"""
            Create HTML for a luxury flight options widget for a trip to {destination} for next week ({date_range_str}).
            The widget should display 2-3 FAKE first-class flight options in a fancy-looking HTML table.
            Table columns should be: Airline, Dates, Details (e.g., 'First Class Suite', 'Flat-bed seat'), Price.
            Use class 'luxury-flight-table' for the table. Add a caption like "First Class Options to {destination}".
            Make the HTML structure elegant and clear, suitable for a luxury theme.
            Do NOT include any CSS or <style> tags. Only provide the raw HTML for the content of the widget.
            """
        else:
            # Potentially handle other styles or return a default message
            widget_html = f"<p>Budget information for '{travel_style}' to {destination} is not yet available.</p>"
            return jsonify({'html': widget_html})

        if prompt_content:
            system_message = Message(role="system", content="You are an HTML generation assistant. You create clean, raw HTML snippets based on user requests. Do not include CSS or <style> tags.")
            user_message = Message(role="user", content=prompt_content)
            
            print(f'[DEBUG] /generate_budget_widget: Sending request to OpenAI for HTML widget.')
            
            try:
                response_data = css_model.invoke(messages=[system_message, user_message]) # Using existing css_model (gpt-3.5-turbo)
                if (
                    response_data and 
                    hasattr(response_data, 'choices') and 
                    response_data.choices and 
                    len(response_data.choices) > 0 and 
                    hasattr(response_data.choices[0], 'message') and 
                    response_data.choices[0].message and 
                    hasattr(response_data.choices[0].message, 'content') and 
                    isinstance(response_data.choices[0].message.content, str)
                ):
                    widget_html = response_data.choices[0].message.content
                    # Basic cleaning: sometimes AI wraps in ```html ... ```
                    widget_html = re.sub(r'^```html\n', '', widget_html)
                    widget_html = re.sub(r'\n```$', '', widget_html)
                    print(f'[DEBUG] /generate_budget_widget: Raw HTML from AI: "{widget_html[:300]}..."')
                else:
                    print("[ERROR] /generate_budget_widget: Failed to generate HTML from AI in expected format.")
                    widget_html = "<p>Error: Could not generate budget details at this time.</p>"
            except Exception as e_ai:
                print(f"[ERROR] /generate_budget_widget: Error during AI call: {e_ai}")
                widget_html = f"<p>Error generating budget details: {str(e_ai)}</p>"
        
        return jsonify({'html': widget_html})

    except Exception as e:
        print(f"[ERROR] /generate_budget_widget: Unexpected error in route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

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

