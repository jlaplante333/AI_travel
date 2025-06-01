from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, g
from flask_socketio import SocketIO # Added for WebSocket support
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
# Removed: from agno.models.core import ChatMessage
from rich.pretty import pprint
from lumaai import LumaAI
import time
from functools import lru_cache
import hashlib
import json
import re
from typing import List, Dict
from dotenv import load_dotenv

# Import TraceCollector
from trace_collector import TraceCollector, AgentTrace, ReasoningStep

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
# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")
# Initialize TraceCollector with Flask app and SocketIO
tracer = TraceCollector(app, socketio)

# Initialize Google Maps client
try:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    # Perform a test geocode, but do not trace it here
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

# Initialize agents (without global scope tracing)
image_agent = Agent(
    model=OpenAIChat(id="gpt-4", api_key=OPENAI_API_KEY),
    tools=[DalleTools()],
    description="AI agent for generating travel images.",
)
print("✅ OpenAI Image Agent Initialized (GPT-4 with DALL-E)")

css_model = OpenAIChat(id="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
print("✅ CSS Generation Agent Initialized (GPT-3.5-Turbo)")


# Serve Frontend
@app.route('/')
def index():
    return render_template('chat.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

@app.route('/itinerary')
def itinerary_page(): # Renamed to avoid conflict with tracer.itinerary
    return render_template('index.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)

# Add timing metrics (can be integrated with tracer later if needed)
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

# Cache (remains unchanged)
ITINERARY_CACHE_SIZE = 100
itinerary_cache = {}
def get_cached_itinerary(cache_key): return itinerary_cache.get(cache_key)
def set_cached_itinerary(cache_key, itinerary_text):
    if len(itinerary_cache) >= ITINERARY_CACHE_SIZE: itinerary_cache.pop(next(iter(itinerary_cache)))
    itinerary_cache[cache_key] = itinerary_text
def generate_cache_key(prompt): return hashlib.md5(prompt.encode()).hexdigest()

def is_valid_destination(destination):
    generic = {"yes", "no", "maybe", "idk", "not sure", "help", "okay", "ok"}
    if not destination or len(destination) < 3: return False
    if destination.lower() in generic: return False
    if destination.isdigit(): return False
    return True

# Generate AI Itinerary using Ollama
@app.route('/generate_itinerary', methods=['POST'])
@tracer.trace_agent('ollama') # Added tracer decorator
def generate_itinerary():
    user_prompt = "Not available" # Default for error logging
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        
        step_input = f"User prompt: {user_prompt}"
        with tracer.trace_step('ollama', 'Validate destination', step_input, 
                              reasoning='Checking if the provided destination is a valid query.') as step:
            destination = user_prompt.strip()
            if not is_valid_destination(destination):
                step.end('error', output='Invalid destination', reasoning='The provided text does not appear to be a valid destination query.')
                return jsonify({"error": "Please enter a valid destination (e.g., 'Paris', 'Tokyo', 'New York')."}), 400
            step.end('success', output=f"Validated destination: {destination}")

        with tracer.trace_step('ollama', 'Geocode destination', destination, 
                              reasoning='Verifying that the destination exists by geocoding it using Google Maps API.') as step:
            if not gmaps:
                step.end('error', output='Google Maps client not available', reasoning='Google Maps client failed to initialize, cannot geocode.')
                return jsonify({"error": "Mapping service unavailable. Cannot validate destination."}), 500
            geocode_result = gmaps.geocode(destination)
            if not geocode_result:
                step.end('error', output=f'"{destination}" not found', reasoning=f'"{destination}" could not be found via geocoding, suggesting it may not be a real destination.')
                return jsonify({"error": f'"{destination}" is not a real destination. Please enter a valid destination (e.g., \'Paris\', \'Tokyo\', \'New York\').'}), 400
            step.end('success', output=f"Geocoded: {geocode_result[0]['formatted_address']}")
        
        os.environ['DESTINATION'] = destination
        print(f"🗺 Generating itinerary for: {user_prompt}")
        
        final_prompt = f"Create a detailed 3-day itinerary for {user_prompt}. Include specific attractions, landmarks, and activities. Format each day with 'Day X:' and list 3-4 activities per day. Make sure to mention specific places to visit."
        with tracer.trace_step('ollama', 'Generate itinerary with Mistral', final_prompt, 
                              reasoning='Sending the structured prompt to Ollama Mistral model to generate a detailed itinerary.') as step:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": final_prompt}],
                options={"temperature": 0.7, "num_predict": 512}
            )
            itinerary_text = response.get("message", {}).get("content", "No itinerary generated.")
            step.end('success', output=itinerary_text[:200] + "..." if itinerary_text else "No itinerary generated.")
        
        return jsonify({'itinerary': itinerary_text, 'destination': destination})
    except Exception as e:
        print(f"❌ Error generating itinerary: {str(e)}")
        active_session = tracer.get_active_session()
        if active_session and active_session.agents and active_session.agents[-1].agentType == 'ollama':
            agent_trace = active_session.agents[-1]
            error_step = ReasoningStep(agent='ollama', action='Itinerary Generation Error', input=user_prompt, output=str(e), reasoning=f'Unhandled exception: {type(e).__name__}', status='error')
            error_step.start_time = time.time() 
            error_step.end()
            agent_trace.add_step(error_step)
        return jsonify({'error': str(e)}), 500

@app.route('/generate_images', methods=['POST'])
@tracer.trace_agent('openai') # Added tracer decorator
def generate_images():
    data = request.json
    location = data.get("location", "default destination")
    
    prompt = f"Create 6 different images of the travel destination, {location}"
    with tracer.trace_step('openai', 'Prepare image generation prompt', location, 
                          reasoning='Creating a detailed prompt for DALL-E image generation based on the location.') as step:
        step.end('success', output=prompt)

    with tracer.trace_step('openai', 'Execute DALL-E image generation', prompt, 
                          reasoning='Running the OpenAI agent with DALL-E tools to generate travel destination images.') as step:
        run_stream = image_agent.run(prompt, stream=True, stream_intermediate_steps=True)
        image_urls = []
        for chunk_idx, chunk in enumerate(run_stream):
            print(f"🔄 OpenAI Chunk {chunk_idx}: {str(chunk)[:100]}...") 
            
            current_reasoning = f"Processing DALL-E stream chunk {chunk_idx}."
            chunk_output = ""

            if hasattr(chunk, "images") and chunk.images:
                for img_idx, img in enumerate(chunk.images):
                    if hasattr(img, "url"):
                        image_urls.append(img.url)
                        chunk_output += f"Image {img_idx+1} URL: {img.url}\n"
                        current_reasoning += f" Extracted image {img_idx+1} URL. "
            if hasattr(chunk, "tools") and chunk.tools:
                for tool_idx, tool in enumerate(chunk.tools):
                    if "content" in tool and "Image has been generated at the URL" in tool["content"]:
                        image_url = tool["content"].split("URL ")[1].strip()
                        image_urls.append(image_url)
                        chunk_output += f"Tool Image {tool_idx+1} URL: {image_url}\n"
                        current_reasoning += f" Extracted tool image {tool_idx+1} URL. "
            
            tracer.record_step(agent_type='openai', action=f'DALL-E Stream Chunk {chunk_idx}', 
                               input_data=f'Chunk {chunk_idx} data', output_data=chunk_output if chunk_output else "Processing...",
                               reasoning=current_reasoning, status='success')

        if not image_urls:
            step.end('error', output='No images generated', reasoning='Failed to generate any images for the destination.')
            return jsonify({"error": "Failed to generate images"}), 500
        
        step.end('success', output=json.dumps({"image_urls_count": len(image_urls)}))
    return jsonify({"image_urls": image_urls})

@app.route('/generate_video', methods=['POST'])
@tracer.trace_agent('lumaai') # Added tracer decorator
def generate_video():
    data = {} # Default for error logging
    try:
        data = request.json
        user_prompt_detail = data.get("prompt", "a 3-day trip exploring scenic landscapes and iconic landmarks")
        
        cinematic_prompt = (
            f"A smooth cinematic travel video showing multiple iconic landmarks and scenic views during a {user_prompt_detail}. "
            "Include famous attractions, local culture, and a beautiful sunset. Multiple camera angles, lively streets, "
            "and relaxing vibes."
        )
        with tracer.trace_step('lumaai', 'Create cinematic prompt', user_prompt_detail, 
                              reasoning='Crafting a detailed cinematic prompt for LumaAI video generation.') as step:
            step.end('success', output=cinematic_prompt)
        
        with tracer.trace_step('lumaai', 'Initialize LumaAI client', 'API Key Check', 
                              reasoning='Initializing the LumaAI client with authentication token.') as step:
            luma_api_key_env = os.getenv("LUMA_API_KEY")
            if not luma_api_key_env:
                step.end('error', output='Luma API Key not found in environment.', reasoning='LUMA_API_KEY environment variable is not set.')
                return jsonify({"error": "LumaAI API Key not configured"}), 500
            client = LumaAI(auth_token=luma_api_key_env)
            step.end('success', output='LumaAI client initialized.')

        with tracer.trace_step('lumaai', 'Submit video generation request', cinematic_prompt, 
                              reasoning='Sending the generation request to LumaAI API.') as step:
            generation = client.generations.create(prompt=cinematic_prompt)
            step.end('success', output=f'Generation ID: {generation.id}')

        with tracer.trace_step('lumaai', 'Monitor generation progress', f'Generation ID: {generation.id}', 
                              reasoning='Polling the LumaAI API to check video generation status.') as outer_poll_step:
            for poll_count in range(20): 
                with tracer.trace_step('lumaai', f'Poll {poll_count+1}/20', f'ID: {generation.id}', reasoning=f'Checking status for generation {generation.id}') as poll_step:
                    generation = client.generations.get(id=generation.id)
                    poll_step.end('success', output=f'State: {generation.state}')

                    if generation.state == "completed":
                        video_url = generation.assets.video
                        print(f"✅ Video ready: {video_url}")
                        outer_poll_step.end('success', output=f'Video URL: {video_url}')
                        return jsonify({"video_url": video_url})
                    elif generation.state == "failed":
                        failure_reason = generation.failure_reason or "Unknown reason"
                        print(f"❌ Generation failed: {failure_reason}")
                        outer_poll_step.end('error', output=f'Failure reason: {failure_reason}')
                        return jsonify({"error": f"Generation failed: {failure_reason}"}), 500
                time.sleep(3)
            
            outer_poll_step.end('error', output='Video generation timed out after 20 polls.')
            return jsonify({"error": "Video generation timed out."}), 500

    except Exception as e:
        print(f"❌ Luma API Exception: {e}")
        active_session = tracer.get_active_session()
        if active_session and active_session.agents and active_session.agents[-1].agentType == 'lumaai':
            agent_trace = active_session.agents[-1]
            error_step = ReasoningStep(agent='lumaai', action='Video Generation Error', input=str(data), output=str(e), reasoning=f'Unhandled exception: {type(e).__name__}', status='error')
            error_step.start_time = time.time(); error_step.end()
            agent_trace.add_step(error_step)
        return jsonify({"error": "Failed to generate video from Luma AI."}), 500

@app.route('/generate_voiceover', methods=['POST'])
@tracer.trace_agent('google-tts') # Added tracer decorator
def generate_voiceover():
    itinerary_text_for_error_logging = "Not available"
    data_for_error_logging = "Not available"
    try:
        with tracer.trace_step('google-tts', 'Parse request data', 'Extracting text from request JSON', 
                              reasoning='Validating and extracting the text content for voiceover.') as step:
            data = request.json
            data_for_error_logging = str(data)
            if not data:
                step.end('error', output='No JSON data provided')
                return jsonify({"error": "No JSON data provided"}), 400
            itinerary_text = data.get("text")
            itinerary_text_for_error_logging = itinerary_text[:100] + "..." if itinerary_text else "No text provided"
            if not itinerary_text:
                step.end('error', output='No text provided in JSON')
                return jsonify({"error": "No text provided in JSON"}), 400
            step.end('success', output=f'Text received (len: {len(itinerary_text)})')

        with tracer.trace_step('google-tts', 'Prepare TTS parameters', itinerary_text[:100] + "...", 
                              reasoning='Setting up voice parameters for natural-sounding speech using Google TTS.') as step:
            client = texttospeech.TextToSpeechClient()
            processed_text = re.sub(r'(Day \d+:)', r'\1<break time="500ms"/>', itinerary_text)
            processed_text = re.sub(r'([A-Z][a-z]+ [A-Z][a-z]+)', r'<emphasis level="moderate">\1</emphasis>', processed_text)
            input_text_obj = texttospeech.SynthesisInput(ssml=f"<speak>{processed_text}</speak>")
            voice_params = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
            audio_config_params = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=0.95, pitch=0)
            step.end('success', output='Voice: en-US Neutral, SSML processed, MP3 output, Rate 0.95')

        with tracer.trace_step('google-tts', 'Generate speech', f'Text length: {len(itinerary_text)} chars', 
                              reasoning='Calling the Google Cloud Text-to-Speech API to synthesize audio.') as step:
            response = client.synthesize_speech(input=input_text_obj, voice=voice_params, audio_config=audio_config_params)
            step.end('success', output=f'Audio content size: {len(response.audio_content)} bytes')

        audio_file_path = "static/ai_voice.mp3"
        with tracer.trace_step('google-tts', 'Save audio file', f'Saving to {audio_file_path}', 
                              reasoning='Writing the generated audio content to an MP3 file for web access.') as step:
            with open(audio_file_path, "wb") as out:
                out.write(response.audio_content)
            step.end('success', output=f'File saved: /{audio_file_path}')
        
        return jsonify({"audio_url": f"/{audio_file_path}"})
    except Exception as e:
        import traceback
        print(f"[DEBUG] /generate_voiceover: !!! ERROR IN ROUTE !!!\n{traceback.format_exc()}")
        active_session = tracer.get_active_session()
        if active_session and active_session.agents and active_session.agents[-1].agentType == 'google-tts':
            agent_trace = active_session.agents[-1]
            error_step = ReasoningStep(agent='google-tts', action='Voiceover Generation Error', input=itinerary_text_for_error_logging, output=str(e), reasoning=f'Unhandled exception: {type(e).__name__}', status='error')
            error_step.start_time = time.time(); error_step.end()
            agent_trace.add_step(error_step)
        return jsonify({"error": f"Failed to generate voiceover: {str(e)}"}), 500

def extract_locations_from_itinerary(itinerary_text: str, base_location: str) -> List[Dict]:
    print(f"[DEBUG] extract_locations_from_itinerary: Called with base_location: '{base_location}'")
    
    prompt_for_extraction = f"""
    Extract all specific location names (attractions, landmarks, restaurants, etc.) from this itinerary.
    Base location: {base_location}
    Itinerary:
    {itinerary_text}
    Return only a JSON array of location names, like:
    ["Location 1", "Location 2", "Location 3"]
    """
    with tracer.trace_step('location', 'Prepare extraction prompt (Ollama)', f"Base: {base_location}, Itinerary len: {len(itinerary_text)}", 
                          reasoning='Creating a prompt for Ollama Mistral to extract location names from the itinerary text.') as step:
        step.end('success', output=prompt_for_extraction[:200] + "...")

    extracted_names = []
    with tracer.trace_step('location', 'Extract location names with Ollama', prompt_for_extraction[:200] + "...", 
                          reasoning='Using Ollama Mistral model to identify and list location names from the itinerary.') as step:
        try:
            response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt_for_extraction}], options={"temperature": 0.3})
            content = response.get("message", {}).get("content", "[]")
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                extracted_names = json.loads(json_match.group())
                step.end('success', output=f'Extracted {len(extracted_names)} names: {str(extracted_names)[:100]}...')
            else:
                step.end('warning', output='No JSON array found in Ollama response.', reasoning='Ollama response did not contain a parsable JSON array of names.')
        except Exception as e:
            step.end('error', output=str(e), reasoning=f'Error during Ollama call for name extraction: {type(e).__name__}')
    
    location_data = []
    if not gmaps:
        tracer.record_step('location', 'Google Maps Client Check', 'Attempting to use gmaps', 'gmaps not initialized', 'Google Maps client (gmaps) not initialized. Skipping geocoding & image search.', 'error')
        return []

    with tracer.trace_step('location', 'Process extracted locations', f'{len(extracted_names)} names found', 
                          reasoning='Iterating through extracted names to geocode and enrich each location.') as outer_step:
        processed_count = 0
        for name_idx, name in enumerate(extracted_names):
            if not isinstance(name, str) or not name.strip(): continue
            location_name = name.strip()
            
            with tracer.trace_step('location', f'Process: {location_name} ({name_idx+1}/{len(extracted_names)})', f'Base: {base_location}', 
                                  reasoning=f'Geocoding and fetching details for "{location_name}".') as loc_step:
                try:
                    search_query_maps = f"{location_name}, {base_location}"
                    geocode_result = gmaps.geocode(search_query_maps)
                    if geocode_result and len(geocode_result) > 0:
                        res = geocode_result[0]
                        lat, lng = res['geometry']['location']['lat'], res['geometry']['location']['lng']
                        description = f"Visit {location_name} in {base_location}."
                        place_id = res.get('place_id')
                        image_url = None
                        
                        tracer.record_step('location', f'Geocoded: {location_name}', search_query_maps, f'Lat: {lat}, Lng: {lng}, PlaceID: {place_id}', 'Successfully geocoded.', 'success')

                        if place_id and GOOGLE_MAPS_API_KEY:
                            with tracer.trace_step('location', f'Fetch Photo: {location_name}', f'Place ID: {place_id}', reasoning='Fetching photo using Google Places API.') as photo_step:
                                try:
                                    place_details = gmaps.place(place_id=place_id, fields=['name', 'photo'])
                                    photos_data = place_details.get('result', {}).get('photos')
                                    if photos_data and len(photos_data) > 0:
                                        photo_reference = photos_data[0].get('photo_reference')
                                        if photo_reference:
                                            image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_MAPS_API_KEY}"
                                            photo_step.end('success', output=f'Image URL created for {location_name}')
                                        else: photo_step.end('warning', output=f'No photo_reference for {location_name}')
                                    else: photo_step.end('warning', output=f'No photos array for {location_name}')
                                except Exception as e_photo:
                                    photo_step.end('error', output=str(e_photo), reasoning=f'Error fetching photo: {type(e_photo).__name__}')
                        
                        location_data.append({"name": location_name, "lat": lat, "lng": lng, "description": description, "image": image_url})
                        processed_count +=1
                        loc_step.end('success', output=f'Processed {location_name}')
                    else:
                        loc_step.end('warning', output=f'Geocoding failed for {location_name}', reasoning='No results from Google Maps Geocoding API.')
                except Exception as e_loc:
                    loc_step.end('error', output=str(e_loc), reasoning=f'Error processing {location_name}: {type(e_loc).__name__}')
        
        outer_step.end('success', output=f'Processed {processed_count}/{len(extracted_names)} locations.')
    return location_data

@app.route('/extract_locations', methods=['POST'])
@tracer.trace_agent('location') # Added tracer decorator
def extract_locations():
    data_for_error_logging = "Not available"
    try:
        with tracer.trace_step('location', 'Parse request', 'Extracting itinerary and location from JSON', 
                              reasoning='Validating and extracting input data for location extraction.') as step:
            data = request.get_json()
            data_for_error_logging = str(data)
            if not data:
                step.end('error', output='No JSON data provided')
                return jsonify({'error': 'Invalid request: No JSON data.'}), 400
            itinerary = data.get('itinerary', '')
            base_location = data.get('location', '')
            if not itinerary:
                step.end('error', output='Itinerary text is required')
                return jsonify({'error': 'Itinerary text is required.'}), 400
            if not base_location:
                base_location = os.getenv('DESTINATION', 'the planned destination')
                step.end('warning', output=f'Base location missing, using fallback: {base_location}', reasoning='Base location not provided, using DESTINATION env var or default.')
            else:
                step.end('success', output=f'Itinerary (len {len(itinerary)}) and location "{base_location}" received.')

        locations = extract_locations_from_itinerary(itinerary_text=itinerary, base_location=base_location)
        return jsonify({'locations': locations})
    except Exception as e:
        import traceback
        print(f"[DEBUG] /extract_locations: !!! UNHANDLED ERROR IN ROUTE !!!\n{traceback.format_exc()}")
        active_session = tracer.get_active_session()
        if active_session and active_session.agents and active_session.agents[-1].agentType == 'location':
            agent_trace = active_session.agents[-1]
            error_step = ReasoningStep(agent='location', action='Location Extraction Error', input=data_for_error_logging, output=str(e), reasoning=f'Unhandled exception: {type(e).__name__}', status='error')
            error_step.start_time = time.time(); error_step.end()
            agent_trace.add_step(error_step)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/randomize_background_css', methods=['POST'])
@tracer.trace_agent('css-generator') # Added tracer decorator
def randomize_background_css():
    data_for_error_logging = "Not available"
    try:
        data = request.get_json()
        data_for_error_logging = str(data)
        theme_prompt_data = data.get('theme_prompt', 'default nature theme')
        
        with tracer.trace_step('css-generator', 'Parse theme prompt', theme_prompt_data, 
                              reasoning='Extracting theme prompt for CSS generation.') as step:
            step.end('success', output=f"Theme: {theme_prompt_data}")

        system_message_content = (
            "You are a CSS generation assistant. The user will provide a theme. "
            "Generate ONLY the CSS rules (properties and values) suitable for the 'body' tag to reflect this theme, focusing on background properties. "
            "Do not include the 'body { ... }\' selector, only the rules. For example, if the theme is \'ocean blue\', "
            "output: background-color: #0077be; background-image: linear-gradient(to bottom, #0077be, #005fa3);"
        )
        
        messages_for_openai = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": theme_prompt_data}
        ]

        with tracer.trace_step('css-generator', 'Generate CSS with OpenAI', f"Theme: {theme_prompt_data}", 
                              reasoning='Sending request to OpenAI GPT-3.5-Turbo for CSS rule generation.') as step:
            response_data_openai = css_model.invoke(messages=messages_for_openai)
            generated_css_rules = ""
            if response_data_openai and hasattr(response_data_openai, 'content') and isinstance(response_data_openai.content, str):
                generated_css_rules = response_data_openai.content.strip()
                if generated_css_rules.startswith('```css'): generated_css_rules = generated_css_rules[len('```css'):].strip()
                if generated_css_rules.endswith('```'): generated_css_rules = generated_css_rules[:-len('```')].strip()
                if generated_css_rules.startswith('{') and generated_css_rules.endswith('}'): generated_css_rules = generated_css_rules[1:-1].strip()
                step.end('success', output=generated_css_rules)
            else:
                step.end('error', output='Failed to generate CSS in expected format.', reasoning=f'OpenAI response was not as expected: {str(response_data_openai)[:100]}')
                return jsonify({'error': 'Failed to generate CSS from model in expected format.'}), 500
        
        css_file_path = os.path.join(app.static_folder, 'generated_background.css')
        full_css_for_file = f'body {{\n    {generated_css_rules}\n}}'
        with tracer.trace_step('css-generator', 'Save CSS to file', css_file_path, 
                              reasoning='Writing the generated CSS rules to a static file.') as step:
            try:
                with open(css_file_path, 'w') as f: f.write(full_css_for_file)
                step.end('success', output=f'CSS saved to {css_file_path}')
            except Exception as e_write:
                step.end('warning', output=str(e_write), reasoning='Error writing CSS to file, but returning CSS to client.')
        
        return jsonify({'css_rules': generated_css_rules})
    except Exception as e:
        import traceback
        print(f'[ERROR] /randomize_background_css: !!! ERROR IN ROUTE !!!\n{traceback.format_exc()}')
        active_session = tracer.get_active_session()
        if active_session and active_session.agents and active_session.agents[-1].agentType == 'css-generator':
            agent_trace = active_session.agents[-1]
            error_step = ReasoningStep(agent='css-generator', action='CSS Generation Error', input=data_for_error_logging, output=str(e), reasoning=f'Unhandled exception: {type(e).__name__}', status='error')
            error_step.start_time = time.time(); error_step.end()
            agent_trace.add_step(error_step)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# WebSocket endpoints for real-time trace streaming
@socketio.on('connect')
def handle_connect():
    print("Client connected to WebSocket")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected from WebSocket")

@socketio.on('subscribe_to_session') 
def handle_subscribe_to_session(data):
    session_id = data.get('session_id')
    if session_id:
        print(f"Client subscribed to trace updates for session {session_id}")
        current_session_data = tracer.export_session(session_id)
        if current_session_data:
            socketio.emit('trace_update', current_session_data, to=request.sid) 

# Run Flask App
if __name__ == '__main__':
    if gmaps:
        print("Google Maps client seems to be initialized and was tested during setup.")
    else:
        print("Warning: Google Maps client (gmaps) was not initialized successfully during setup.")
    
    print("🚀 Starting Flask app with SocketIO and TraceCollector...")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)
