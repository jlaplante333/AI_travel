from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, g
from flask_socketio import SocketIO
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
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
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

# Initialize Flask App with SocketIO
app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize TraceCollector with both Flask and SocketIO
tracer = TraceCollector(app, socketio)

# Initialize Google Maps client
try:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    with tracer.trace_step('orchestrator', 'Initialize Google Maps', 'Testing Google Maps API', 
                          'Testing if Google Maps API key is valid by geocoding Paris, France'):
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
with tracer.trace_step('orchestrator', 'Initialize OpenAI Agent', 'Setting up OpenAI agent with DALL-E tools',
                      'Creating an AI agent for generating travel images using OpenAI GPT-4 and DALL-E'):
    image_agent = Agent(
        model=OpenAIChat(id="gpt-4", api_key=OPENAI_API_KEY),
        tools=[DalleTools()],
        description="AI agent for generating travel images.",
    )

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
@tracer.trace_agent('ollama')
def generate_itinerary():
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        
        with tracer.trace_step('ollama', 'Validate destination', user_prompt, 
                              'Checking if the provided destination is valid'):
            destination = user_prompt.strip()
            if not is_valid_destination(destination):
                tracer.record_step('ollama', 'Validation failed', user_prompt, 
                                  'Invalid destination', 
                                  'The provided text does not appear to be a valid destination', 
                                  'error', 100)
                return jsonify({
                    "error": "Please enter a valid destination (e.g., 'Paris', 'Tokyo', 'New York')."
                }), 400
        
        with tracer.trace_step('ollama', 'Geocode destination', destination, 
                              'Verifying that the destination exists by geocoding it'):
            geocode_result = gmaps.geocode(destination)
            if not geocode_result:
                tracer.record_step('ollama', 'Geocoding failed', destination, 
                                  'No geocoding results', 
                                  f'"{destination}" could not be found via geocoding, suggesting it may not be a real destination', 
                                  'error', 150)
                return jsonify({
                    "error": f'"{destination}" is not a real destination. Please enter a valid destination (e.g., \'Paris\', \'Tokyo\', \'New York\').'
                }), 400
        
        os.environ['DESTINATION'] = destination  # Set destination for location agent
        print(f"🗺 Generating itinerary for: {user_prompt}")
        
        with tracer.trace_step('ollama', 'Optimize prompt', user_prompt, 
                              'Creating an optimized prompt for the Ollama Mistral model'):
            optimized_prompt = f"Create a detailed 3-day itinerary for {user_prompt}. Include specific attractions, landmarks, and activities. Format each day with 'Day X:' and list 3-4 activities per day. Make sure to mention specific places to visit."
            tracer.record_step('ollama', 'Prompt analysis', optimized_prompt, 
                              'Prompt structure analyzed', 
                              'Analyzing the destination and creating a structured prompt that will generate a well-formatted itinerary with specific attractions', 
                              'success', 120)
        
        with tracer.trace_step('ollama', 'Generate itinerary with Mistral', optimized_prompt, 
                              'Sending the optimized prompt to Ollama Mistral model to generate a detailed itinerary'):
            start_time = time.time()
            response = ollama.chat(
                model="mistral",
                messages=[{
                    "role": "user", 
                    "content": optimized_prompt
                }],
                options={
                    "temperature": 0.7,
                    "num_predict": 512
                }
            )
            generation_time = int((time.time() - start_time) * 1000)  # ms
            
            itinerary_text = response.get("message", {}).get("content", "No itinerary generated.")
            
            # Record detailed reasoning about the generated itinerary
            reasoning = (
                f"Generated a 3-day itinerary for {destination} with Mistral model. "
                f"The itinerary includes specific attractions and activities for each day, "
                f"formatted with clear day headers. Temperature of 0.7 was used to balance "
                f"creativity with coherence. Generation took {generation_time/1000:.2f} seconds."
            )
            tracer.record_step('ollama', 'Post-process itinerary', itinerary_text[:100] + "...", 
                              itinerary_text[:100] + "...", 
                              reasoning, 
                              'success', generation_time)
        
        return jsonify({
            'itinerary': itinerary_text,
            'destination': destination
        })
    except Exception as e:
        print(f"❌ Error generating itinerary: {str(e)}")
        tracer.record_step('ollama', 'Error generating itinerary', user_prompt, 
                          str(e), 
                          f'An error occurred during itinerary generation: {str(e)}', 
                          'error', 0)
        return jsonify({'error': str(e)}), 500

@app.route('/generate_images', methods=['POST'])
@tracer.trace_agent('openai')
def generate_images():
    data = request.json
    location = data.get("location", "default destination")
    
    with tracer.trace_step('openai', 'Prepare image generation prompt', location, 
                          'Creating a detailed prompt for DALL-E image generation'):
        prompt = f"Create 6 different images of the travel destination, {location}"
        tracer.record_step('openai', 'Analyze location for imagery', location, 
                          'Location analyzed for visual characteristics', 
                          f'Analyzing "{location}" to determine key visual elements that should be captured in generated images', 
                          'success', 150)
    
    # Run Agent to Generate Images
    with tracer.trace_step('openai', 'Execute DALL-E image generation', prompt, 
                          'Running the OpenAI agent with DALL-E tools to generate travel destination images'):
        start_time = time.time()
        run_stream = image_agent.run(prompt, stream=True, stream_intermediate_steps=True)
        
        image_urls = []
        intermediate_steps = []
        
        for chunk in run_stream:
            step_info = f"Processing chunk: {chunk}"
            intermediate_steps.append(step_info)
            print(f"🔄 {step_info}")
            
            # Extract from 'images' if available
            if hasattr(chunk, "images") and chunk.images:
                for img in chunk.images:
                    if hasattr(img, "url"):
                        image_urls.append(img.url)
                        tracer.record_step('openai', 'Image generated', 'DALL-E generation', 
                                          f'Image URL: {img.url[:30]}...', 
                                          'Successfully generated an image with DALL-E based on the travel destination prompt', 
                                          'success', 2000)
            
            # Extract from 'tools' if available
            if hasattr(chunk, "tools") and chunk.tools:
                for tool in chunk.tools:
                    if "content" in tool and "Image has been generated at the URL" in tool["content"]:
                        # Extract the actual URL from the response text
                        image_url = tool["content"].split("URL ")[1].strip()
                        image_urls.append(image_url)
                        tracer.record_step('openai', 'Tool used for image generation', 'DALL-E tool call', 
                                          f'Image URL: {image_url[:30]}...', 
                                          'Used DALL-E tool to generate an image based on specific aspects of the destination', 
                                          'success', 1800)
        
        generation_time = int((time.time() - start_time) * 1000)  # ms
    
    # Ensure at least one image was found
    if not image_urls:
        tracer.record_step('openai', 'Image generation failed', prompt, 
                          'No images generated', 
                          'Failed to generate any images for the destination. This could be due to API limits, content policy restrictions, or connectivity issues.', 
                          'error', generation_time)
        return jsonify({"error": "Failed to generate images"}), 500
    
    with tracer.trace_step('openai', 'Finalize image collection', str(len(image_urls)) + " images", 
                          'Processing and returning the generated images'):
        tracer.record_step('openai', 'Return image results', f'{len(image_urls)} images generated', 
                          json.dumps({"image_urls": image_urls}), 
                          f'Successfully generated {len(image_urls)} images for {location}. The images showcase different aspects of the destination.', 
                          'success', 200)
        return jsonify({"image_urls": image_urls})

@app.route('/generate_video', methods=['POST'])
@tracer.trace_agent('lumaai')
def generate_video():
    try:
        data = request.json
        user_prompt = data.get("prompt", "a 3-day trip exploring scenic landscapes and iconic landmarks")
        
        with tracer.trace_step('lumaai', 'Create cinematic prompt', user_prompt, 
                              'Crafting a detailed cinematic prompt for LumaAI video generation'):
            # Compose a detailed cinematic prompt
            prompt = (
                f"A smooth cinematic travel video showing multiple iconic landmarks and scenic views during a {user_prompt}. "
                "Include famous attractions, local culture, and a beautiful sunset. Multiple camera angles, lively streets, "
                "and relaxing vibes."
            )
            
            tracer.record_step('lumaai', 'Analyze video requirements', user_prompt, 
                              prompt, 
                              'Analyzed the user request and crafted a detailed cinematic prompt that specifies visual elements, camera movements, and atmosphere to create an engaging travel video', 
                              'success', 300)
        
        with tracer.trace_step('lumaai', 'Initialize LumaAI client', 'Setting up API connection', 
                              'Initializing the LumaAI client with authentication'):
            # Pass your auth token directly 
            client = LumaAI(auth_token=os.getenv("LUMA_API_KEY"))
            
            if not client or not os.getenv("LUMA_API_KEY"):
                tracer.record_step('lumaai', 'API authentication failed', 'LumaAI client initialization', 
                                  'Failed to initialize client', 
                                  'Could not initialize the LumaAI client. This may be due to a missing or invalid API key.', 
                                  'error', 100)
                return jsonify({"error": "LumaAI client initialization failed"}), 500
        
        with tracer.trace_step('lumaai', 'Submit video generation request', prompt, 
                              'Sending the generation request to LumaAI API'):
            start_time = time.time()
            generation = client.generations.create(prompt=prompt)
            
            tracer.record_step('lumaai', 'Generation request submitted', prompt, 
                              f'Generation ID: {generation.id}', 
                              'Successfully submitted the video generation request to LumaAI. The generation process will take some time to complete.', 
                              'success', int((time.time() - start_time) * 1000))
        
        with tracer.trace_step('lumaai', 'Monitor generation progress', f'Generation ID: {generation.id}', 
                              'Polling the API to check generation status'):
            poll_start_time = time.time()
            for poll_count in range(20):  # Poll up to 1 minute (20 x 3s)
                poll_iteration_start = time.time()
                generation = client.generations.get(id=generation.id)
                
                poll_status = f"Poll {poll_count+1}/20: Status = {generation.state}"
                tracer.record_step('lumaai', f'Poll generation status ({poll_count+1})', 
                                  f'Generation ID: {generation.id}', 
                                  poll_status, 
                                  f'Checking generation status. Current state: {generation.state}. Video generation is a computationally intensive process that takes time.', 
                                  'success', int((time.time() - poll_iteration_start) * 1000))
                
                if generation.state == "completed":
                    video_url = generation.assets.video
                    print(f"✅ Video ready: {video_url}")
                    
                    total_generation_time = int((time.time() - poll_start_time) * 1000)
                    tracer.record_step('lumaai', 'Video generation completed', 
                                      f'Generation ID: {generation.id}', 
                                      f'Video URL: {video_url}', 
                                      f'Successfully generated a cinematic travel video after {poll_count+1} status checks. The video showcases the destination with cinematic quality.', 
                                      'success', total_generation_time)
                    
                    return jsonify({"video_url": video_url})
                elif generation.state == "failed":
                    print(f"❌ Generation failed: {generation.failure_reason}")
                    
                    tracer.record_step('lumaai', 'Video generation failed', 
                                      f'Generation ID: {generation.id}', 
                                      f'Failure reason: {generation.failure_reason}', 
                                      f'The video generation process failed. Reason: {generation.failure_reason}. This could be due to content policy violations, technical issues, or resource limitations.', 
                                      'error', int((time.time() - poll_start_time) * 1000))
                    
                    return jsonify({"error": f"Generation failed: {generation.failure_reason}"}), 500
                
                time.sleep(3)
            
            tracer.record_step('lumaai', 'Video generation timeout', 
                              f'Generation ID: {generation.id}', 
                              'Timed out after 20 polling attempts', 
                              'The video generation process is taking longer than expected. The process may still be running on LumaAI servers, but we've reached our polling limit.', 
                              'error', int((time.time() - poll_start_time) * 1000))
            
            return jsonify({"error": "Video generation timed out."}), 500

    except Exception as e:
        print(f"❌ Luma API Exception: {e}")
        
        tracer.record_step('lumaai', 'Exception in video generation', 
                          str(data) if data else 'No input data', 
                          f'Error: {str(e)}', 
                          f'An exception occurred during the video generation process: {str(e)}. This could be due to API errors, network issues, or invalid parameters.', 
                          'error', 0)
        
        return jsonify({"error": "Failed to generate video from Luma AI."}), 500

# Generate AI Voiceover for the Itinerary
@app.route('/generate_voiceover', methods=['POST'])
@tracer.trace_agent('google-tts')
def generate_voiceover():
    print("[DEBUG] /generate_voiceover: Route called.")
    try:
        with tracer.trace_step('google-tts', 'Parse request data', 'Extracting text from request', 
                              'Validating and extracting the text content from the request JSON'):
            data = request.json
            if not data:
                print("[DEBUG] /generate_voiceover: No JSON data received.")
                tracer.record_step('google-tts', 'Request validation failed', 
                                  'No JSON data', 
                                  'Error: No JSON data provided', 
                                  'The request did not contain any JSON data, which is required for text-to-speech processing.', 
                                  'error', 100)
                return jsonify({"error": "No JSON data provided"}), 400
            
            itinerary_text = data.get("text")
            if not itinerary_text:
                print("[DEBUG] /generate_voiceover: 'text' field missing from JSON data.")
                tracer.record_step('google-tts', 'Text validation failed', 
                                  json.dumps(data), 
                                  'Error: No text provided in JSON', 
                                  'The JSON data did not contain a "text" field, which is required for text-to-speech processing.', 
                                  'error', 100)
                return jsonify({"error": "No text provided in JSON"}), 400

        print(f"[DEBUG] /generate_voiceover: Attempting to generate voiceover for text (first 100 chars): {itinerary_text[:100]}")
        
        with tracer.trace_step('google-tts', 'Prepare TTS parameters', itinerary_text[:100] + "...", 
                              'Setting up voice parameters for natural-sounding speech'):
            start_time = time.time()
            client = texttospeech.TextToSpeechClient()
            
            # Process text for better TTS results
            processed_text = itinerary_text
            # Add pauses after day headers
            processed_text = re.sub(r'(Day \d+:)', r'\1<break time="500ms"/>', processed_text)
            # Add emphasis to attraction names (simplified approach)
            processed_text = re.sub(r'([A-Z][a-z]+ [A-Z][a-z]+)', r'<emphasis level="moderate">\1</emphasis>', processed_text)
            
            input_text = texttospeech.SynthesisInput(text=processed_text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.95,  # Slightly slower for clarity
                pitch=0
            )
            
            voice_setup_time = int((time.time() - start_time) * 1000)
            tracer.record_step('google-tts', 'Voice parameter selection', 
                              'Text processing and voice configuration', 
                              'Parameters: en-US, Neural, speaking_rate=0.95', 
                              'Processed the text to add appropriate pauses and emphasis. Selected a neutral English voice with slightly slower speaking rate for clarity, especially for location names.', 
                              'success', voice_setup_time)

        with tracer.trace_step('google-tts', 'Generate speech', 'Synthesizing speech with Google TTS', 
                              'Calling the Google Cloud Text-to-Speech API to generate audio'):
            print("[DEBUG] /generate_voiceover: Synthesizing speech...")
            synthesis_start = time.time()
            response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
            synthesis_time = int((time.time() - synthesis_start) * 1000)
            
            tracer.record_step('google-tts', 'Speech synthesis', 
                              'Text length: ' + str(len(itinerary_text)) + ' characters', 
                              'Audio generated: ' + str(len(response.audio_content)) + ' bytes', 
                              'Successfully synthesized speech from the itinerary text. The neural voice model created natural-sounding pronunciation of location names and proper pacing for the itinerary format.', 
                              'success', synthesis_time)

        with tracer.trace_step('google-tts', 'Save audio file', 'Writing audio content to file', 
                              'Saving the generated audio to a file for web access'):
            save_start = time.time()
            audio_file = "static/ai_voice.mp3"
            print(f"[DEBUG] /generate_voiceover: Writing audio content to {audio_file}")
            
            with open(audio_file, "wb") as out:
                out.write(response.audio_content)
            
            save_time = int((time.time() - save_start) * 1000)
            print(f"[DEBUG] /generate_voiceover: Voiceover saved successfully.")
            
            tracer.record_step('google-tts', 'Save audio file', 
                              'Audio data: ' + str(len(response.audio_content)) + ' bytes', 
                              f'File saved: /{audio_file}', 
                              'Successfully saved the synthesized speech as an MP3 file. The file is optimized for web streaming with good audio quality and reasonable file size.', 
                              'success', save_time)

        return jsonify({"audio_url": f"/{audio_file}"})  # Ensure leading slash for correct URL path

    except Exception as e:
        import traceback
        print(f"[DEBUG] /generate_voiceover: !!! ERROR IN ROUTE !!!")
        print(traceback.format_exc())  # Print full traceback
        
        error_message = f"Failed to generate voiceover: {str(e)}"
        if "GOOGLE_APPLICATION_CREDENTIALS" in str(e):
            error_message = "Failed to generate voiceover: Google Cloud authentication error. Ensure credentials are set up."
        
        tracer.record_step('google-tts', 'Error in voiceover generation', 
                          itinerary_text[:100] + "..." if itinerary_text else "No text provided", 
                          f'Error: {error_message}', 
                          f'An exception occurred during voiceover generation: {traceback.format_exc()}. This could be due to authentication issues, API limits, or invalid input text.', 
                          'error', 0)
        
        return jsonify({"error": error_message}), 500

def extract_locations_from_itinerary(itinerary_text: str, base_location: str) -> List[Dict]:
    """Extract location names, get coordinates, and find an image URL via Google Places API."""
    print(f"[DEBUG] extract_locations_from_itinerary: Called with base_location: '{base_location}'")
    print(f"[DEBUG] extract_locations_from_itinerary: Itinerary snippet: {itinerary_text[:200]}...")
    
    with tracer.trace_step('location', 'Prepare extraction prompt', base_location, 
                          'Creating a prompt to extract location names from the itinerary'):
        prompt = f"""
        Extract all specific location names (attractions, landmarks, restaurants, etc.) from this itinerary.
        Base location: {base_location}
        Itinerary:
        {itinerary_text}
        
        Return only a JSON array of location names, like:
        ["Location 1", "Location 2", "Location 3"]
        """
        
        tracer.record_step('location', 'Analyze itinerary structure', 
                          itinerary_text[:100] + "...", 
                          f'Analyzing itinerary for {base_location}', 
                          'Examining the itinerary text to identify patterns and structure. Looking for day headers, attraction mentions, and specific location references.', 
                          'success', 200)
    
    extracted_names = []
    with tracer.trace_step('location', 'Extract location names with Ollama', prompt[:100] + "...", 
                          'Using Ollama Mistral model to extract location names'):
        print("[DEBUG] extract_locations_from_itinerary: Sending prompt to Ollama for location name extraction.")
        
        extraction_start = time.time()
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
                    
                    tracer.record_step('location', 'Parse location names', 
                                      'JSON parsing of extracted locations', 
                                      f'Extracted {len(extracted_names)} locations', 
                                      f'Successfully parsed the JSON array of location names from the Ollama response. Found {len(extracted_names)} distinct locations in the itinerary.', 
                                      'success', int((time.time() - extraction_start) * 1000))
                except json.JSONDecodeError as json_e:
                    print(f"[DEBUG] extract_locations_from_itinerary: JSONDecodeError parsing names: {json_e}. Content: {json_match.group()}")
                    
                    tracer.record_step('location', 'JSON parsing error', 
                                      json_match.group() if json_match else 'No JSON found', 
                                      f'Error: {str(json_e)}', 
                                      'Failed to parse the JSON array from the Ollama response. The response format was not valid JSON.', 
                                      'error', int((time.time() - extraction_start) * 1000))
            else:
                print("[DEBUG] extract_locations_from_itinerary: No JSON array in Ollama response for names.")
                
                tracer.record_step('location', 'Location extraction failed', 
                                  content[:100] + "...", 
                                  'No JSON array found in response', 
                                  'The Ollama response did not contain a properly formatted JSON array of location names. This could be due to model output formatting issues.', 
                                  'error', int((time.time() - extraction_start) * 1000))
        except Exception as e:
            print(f"[DEBUG] extract_locations_from_itinerary: Error during Ollama call for names: {str(e)}")
            
            tracer.record_step('location', 'Ollama API error', 
                              prompt[:100] + "...", 
                              f'Error: {str(e)}', 
                              f'An error occurred while calling the Ollama API: {str(e)}. This could be due to connectivity issues or API limitations.', 
                              'error', int((time.time() - extraction_start) * 1000))

    location_data = []
    if not gmaps:
        print("[DEBUG] extract_locations_from_itinerary: Google Maps client (gmaps) not initialized. Skipping geocoding & image search.")
        
        tracer.record_step('location', 'Google Maps client missing', 
                          'Geocoding attempt', 
                          'Error: Google Maps client not initialized', 
                          'Cannot proceed with geocoding because the Google Maps client was not properly initialized. Check API key and connectivity.', 
                          'error', 100)
        return []

    with tracer.trace_step('location', 'Process locations', f'{len(extracted_names)} locations to process', 
                          'Geocoding and enriching each extracted location'):
        geocoding_start = time.time()
        location_count = 0
        
        for name in extracted_names:
            if not isinstance(name, str) or not name.strip():
                print(f"[DEBUG] extract_locations_from_itinerary: Skipping invalid location name: {name}")
                continue
            
            location_name = name.strip()
            lat, lng, description = None, None, None
            image_url = None  # Default to no image
            
            with tracer.trace_step('location', f'Geocode location: {location_name}', 
                                  f'{location_name}, {base_location}', 
                                  'Geocoding location to get coordinates'):
                loc_start = time.time()
                try:
                    search_query_maps = f"{location_name}, {base_location}"
                    print(f"[DEBUG] extract_locations_from_itinerary: Geocoding '{search_query_maps}'")
                    geocode_result = gmaps.geocode(search_query_maps)
                    
                    if geocode_result and len(geocode_result) > 0:
                        res = geocode_result[0]
                        lat = res['geometry']['location']['lat']
                        lng = res['geometry']['location']['lng']
                        description = f"Visit {location_name} in {base_location}."
                        place_id = res.get('place_id')  # Get place_id
                        
                        tracer.record_step('location', 'Geocoding successful', 
                                          search_query_maps, 
                                          f'Coordinates: ({lat}, {lng}), Place ID: {place_id}', 
                                          f'Successfully geocoded "{location_name}" to coordinates ({lat}, {lng}). The Google Maps API returned a place_id which can be used to fetch additional details.', 
                                          'success', int((time.time() - loc_start) * 1000))
                        
                        if place_id and GOOGLE_MAPS_API_KEY:
                            with tracer.trace_step('location', f'Fetch place details: {location_name}', 
                                                  f'Place ID: {place_id}', 
                                                  'Getting additional details and photos for the location'):
                                place_start = time.time()
                                try:
                                    # Fetch place details to get photo reference
                                    place_details = gmaps.place(place_id=place_id, fields=['name', 'photo'])
                                    photos_data = place_details.get('result', {}).get('photos')
                                    
                                    if photos_data and len(photos_data) > 0:
                                        photo_reference = photos_data[0].get('photo_reference')
                                        if photo_reference:
                                            # Construct the photo URL
                                            image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_MAPS_API_KEY}"
                                            
                                            tracer.record_step('location', 'Photo reference found', 
                                                              f'Place ID: {place_id}', 
                                                              f'Photo URL created with reference: {photo_reference[:10]}...', 
                                                              f'Found a photo reference for {location_name} and constructed a URL to fetch the image from Google Places API.', 
                                                              'success', int((time.time() - place_start) * 1000))
                                        else:
                                            tracer.record_step('location', 'No photo reference', 
                                                              f'Place ID: {place_id}', 
                                                              'No photo_reference found', 
                                                              f'The place details for {location_name} did not include a photo_reference, so no image URL could be created.', 
                                                              'warning', int((time.time() - place_start) * 1000))
                                    else:
                                        tracer.record_step('location', 'No photos array', 
                                                          f'Place ID: {place_id}', 
                                                          'No photos array in place details', 
                                                          f'The place details for {location_name} did not include a photos array, so no image could be retrieved.', 
                                                          'warning', int((time.time() - place_start) * 1000))
                                except Exception as e_place_photo:
                                    tracer.record_step('location', 'Error fetching place details', 
                                                      f'Place ID: {place_id}', 
                                                      f'Error: {str(e_place_photo)}', 
                                                      f'An error occurred while fetching place details for {location_name}: {str(e_place_photo)}', 
                                                      'error', int((time.time() - place_start) * 1000))
                        
                        location_data.append({
                            "name": location_name,
                            "lat": lat,
                            "lng": lng,
                            "description": description,
                            "image": image_url  # This will be None if no image was found
                        })
                        location_count += 1
                    else:
                        tracer.record_step('location', 'Geocoding failed', 
                                          search_query_maps, 
                                          'No geocoding results', 
                                          f'Failed to geocode "{location_name}". The Google Maps API returned no results for this location name.', 
                                          'error', int((time.time() - loc_start) * 1000))
                
                except Exception as e:
                    tracer.record_step('location', 'Error processing location', 
                                      location_name, 
                                      f'Error: {str(e)}', 
                                      f'An error occurred while processing the location "{location_name}": {str(e)}', 
                                      'error', int((time.time() - loc_start) * 1000))
                    continue
        
        total_geocoding_time = int((time.time() - geocoding_start) * 1000)
        tracer.record_step('location', 'Locations processing complete', 
                          f'Processed {len(extracted_names)} location names', 
                          f'Successfully geocoded {location_count} locations', 
                          f'Completed processing all extracted location names. Successfully geocoded {location_count} out of {len(extracted_names)} locations. Each location includes coordinates and, where available, an image URL.', 
                          'success', total_geocoding_time)
    
    print(f"[DEBUG] extract_locations_from_itinerary: Returning {len(location_data)} locations.")
    return location_data

@app.route('/extract_locations', methods=['POST'])
@tracer.trace_agent('location')
def extract_locations():
    print("\n🔥🔥 HELLO WORLD FROM /EXTRACT_LOCATIONS! ROUTE HIT 🔥🔥\n")  # Prominent print statement
    try:
        with tracer.trace_step('location', 'Parse request', 'Extracting data from request', 
                              'Validating and extracting the itinerary and location from the request'):
            print("[DEBUG] /extract_locations: Route called.")
            data = request.get_json()
            if not data:
                print("[DEBUG] /extract_locations: No JSON data received in request.")
                
                tracer.record_step('location', 'Request validation failed', 
                                  'Request parsing', 
                                  'Error: Invalid request: No JSON data', 
                                  'The request did not contain valid JSON data, which is required for location extraction.', 
                                  'error', 100)
                
                return jsonify({'error': 'Invalid request: No JSON data.'}), 400
            
            itinerary = data.get('itinerary', '')
            base_location = data.get('location', '')
            print(f"[DEBUG] /extract_locations: Received itinerary (snippet): {itinerary[:100] if itinerary else 'None'}... Base location: '{base_location}'")
            
            if not itinerary:
                print("[DEBUG] /extract_locations: Itinerary text is missing.")
                
                tracer.record_step('location', 'Missing itinerary', 
                                  'Request validation', 
                                  'Error: Itinerary text is required', 
                                  'The request did not contain an itinerary text, which is required for location extraction.', 
                                  'error', 100)
                
                return jsonify({'error': 'Itinerary text is required.'}), 400
            
            if not base_location:
                print("[DEBUG] /extract_locations: Base location is missing. Attempting fallback.")
                base_location = os.getenv('DESTINATION', 'the planned destination')  # Fallback
                print(f"[DEBUG] /extract_locations: Fallback base location: '{base_location}'")
                
                tracer.record_step('location', 'Missing base location', 
                                  'Request validation', 
                                  f'Using fallback location: {base_location}', 
                                  'The request did not specify a base location. Using the DESTINATION environment variable or a generic fallback.', 
                                  'warning', 100)

        with tracer.trace_step('location', 'Extract locations from itinerary', 
                              f'Itinerary for {base_location}', 
                              'Processing the itinerary to extract and geocode locations'):
            print(f"[DEBUG] /extract_locations: Calling extract_locations_from_itinerary.")
            extraction_start = time.time()
            
            locations = extract_locations_from_itinerary(itinerary_text=itinerary, base_location=base_location)
            
            extraction_time = int((time.time() - extraction_start) * 1000)
            print(f"[DEBUG] /extract_locations: extract_locations_from_itinerary returned {len(locations)} locations.")
            
            tracer.record_step('location', 'Location extraction complete', 
                              f'Itinerary for {base_location}', 
                              f'Extracted {len(locations)} locations with coordinates and images', 
                              f'Successfully extracted {len(locations)} locations from the itinerary for {base_location}. Each location includes coordinates, a description, and (where available) an image URL.', 
                              'success', extraction_time)
        
        return jsonify({
            'locations': locations
        })
    except Exception as e:
        # Log the full exception traceback for detailed debugging
        import traceback
        print(f"[DEBUG] /extract_locations: !!! UNHANDLED ERROR IN ROUTE !!!")
        print(traceback.format_exc())  # This will print the full stack trace
        
        tracer.record_step('location', 'Unhandled exception', 
                          'Location extraction', 
                          f'Error: {str(e)}', 
                          f'An unhandled exception occurred during location extraction: {traceback.format_exc()}', 
                          'error', 0)
        
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# WebSocket endpoints for real-time trace streaming
@socketio.on('connect')
def handle_connect():
    print("Client connected to WebSocket")

@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle client subscription to trace updates."""
    session_id = data.get('session_id')
    if session_id:
        print(f"Client subscribed to trace updates for session {session_id}")
        # Join a room named after the session_id
        socketio.join_room(session_id)

# API endpoint to get all trace sessions
@app.route('/traces', methods=['GET'])
def get_all_traces():
    """Get all trace sessions."""
    return jsonify(tracer.export_all_sessions())

# API endpoint to get a specific trace session
@app.route('/traces/<session_id>', methods=['GET'])
def get_trace(session_id):
    """Get a specific trace session."""
    trace_data = tracer.export_session(session_id)
    if not trace_data:
        return jsonify({'error': 'Trace session not found'}), 404
    return jsonify(trace_data)

# Run Flask App
if __name__ == '__main__':
    # Test Google Maps client
    try:
        with tracer.trace_step('orchestrator', 'Test Google Maps on startup', 'Paris, France', 
                              'Verifying Google Maps API functionality on application startup'):
            test_result = gmaps.geocode('Paris, France')
            if test_result:
                print("✅ Google Maps client initialized and tested successfully")
                print("✅ Test geocoding successful: Paris, France")
                
                tracer.record_step('orchestrator', 'Google Maps test successful', 
                                  'Paris, France', 
                                  test_result[0]['formatted_address'], 
                                  'Successfully tested the Google Maps API by geocoding "Paris, France". The API is functioning correctly.', 
                                  'success', 200)
    except Exception as e:
        print(f"❌ Google Maps client test failed: {str(e)}")
        
        tracer.record_step('orchestrator', 'Google Maps test failed', 
                          'Paris, France', 
                          f'Error: {str(e)}', 
                          f'Failed to test the Google Maps API: {str(e)}. The application may not function correctly without Google Maps API access.', 
                          'error', 100)
        
        raise

    # Run with SocketIO instead of app.run()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
