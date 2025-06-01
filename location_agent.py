import re
from typing import List, Dict, Optional
import googlemaps
from dataclasses import dataclass
import os
from dotenv import load_dotenv

@dataclass
class Location:
    name: str
    description: str
    lat: float
    lng: float
    image: Optional[str] = None

class LocationAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")
        self.gmaps = googlemaps.Client(key=api_key)

    def extract_locations_from_itinerary(self, itinerary: str) -> List[Location]:
        print("LocationAgent: Starting to extract locations from itinerary text...")
        days = re.split(r'Day \d+:', itinerary)
        locations = []
        seen_locations = set()  # Track unique locations
        
        for day in days:
            location_patterns = [
                r'(?:Visit|Explore|See|Tour|Stop by|Head to|Go to|Check out)\s+([^,.!?]+)',
                r'(?:at|in)\s+([^,.!?]+(?:Museum|Park|Palace|Castle|Cathedral|Square|Market|Garden|Tower|Bridge|Center|Hall|Theater|Opera|Gallery|Zoo|Aquarium|Beach|Mountain|Lake|River|Valley|District|Quarter|Street|Avenue|Boulevard|Plaza|Stadium|Arena|Monument|Memorial|Statue|Temple|Church|Mosque|Synagogue|Shrine|Fort|Ruins|Reserve|Sanctuary|Observatory|Observatory|Planetarium|Botanical|Zoo|Aquarium|Beach|Mountain|Lake|River|Valley|District|Quarter|Street|Avenue|Boulevard|Plaza|Stadium|Arena|Monument|Memorial|Statue|Temple|Church|Mosque|Synagogue|Shrine|Fort|Ruins|Reserve|Sanctuary|Observatory|Planetarium|Botanical))',
                r'(?:the|a|an)\s+([^,.!?]+(?:Museum|Park|Palace|Castle|Cathedral|Square|Market|Garden|Tower|Bridge|Center|Hall|Theater|Opera|Gallery|Zoo|Aquarium|Beach|Mountain|Lake|River|Valley|District|Quarter|Street|Avenue|Boulevard|Plaza|Stadium|Arena|Monument|Memorial|Statue|Temple|Church|Mosque|Synagogue|Shrine|Fort|Ruins|Reserve|Sanctuary|Observatory|Planetarium|Botanical))',
                r'(?:famous|popular|iconic|historic|beautiful|stunning|magnificent|ancient|medieval|modern)\s+([^,.!?]+)'
            ]
            
            for pattern in location_patterns:
                location_matches = re.finditer(pattern, day, re.IGNORECASE)
                
                for match in location_matches:
                    location_name = match.group(1).strip()
                    if location_name.lower() in seen_locations:
                        continue
                    sentence_start = max(0, day.find(location_name) - 100)
                    sentence_end = min(len(day), day.find(location_name) + 100)
                    description = day[sentence_start:sentence_end].strip()
                    location_name = re.sub(r'^(the|a|an)\s+', '', location_name, flags=re.IGNORECASE)
                    location_name = location_name.strip()
                    if len(location_name) < 3 or location_name.lower() in ['city', 'town', 'place', 'area', 'region']:
                        continue
                    try:
                        destination = os.getenv('DESTINATION', '')
                        search_query = f"{location_name}, {destination}" if destination else location_name
                        print(f"LocationAgent: Searching for coordinates of '{search_query}'...")
                        geocode_result = self.gmaps.geocode(search_query)
                        if geocode_result:
                            formatted_address = geocode_result[0]['formatted_address']
                            if destination and destination.lower() not in formatted_address.lower():
                                print(f"LocationAgent: Skipping {location_name} – not in {destination} (address: {formatted_address})")
                                continue
                            lat = geocode_result[0]['geometry']['location']['lat']
                            lng = geocode_result[0]['geometry']['location']['lng']
                            print(f"LocationAgent: Found coordinates for '{location_name}': (lat, lng) = ({lat}, {lng})")
                            location = Location(name=location_name, description=description, lat=lat, lng=lng)
                            locations.append(location)
                            seen_locations.add(location_name.lower())
                        else:
                            print(f"LocationAgent: No coordinates found for '{search_query}'")
                    except Exception as e:
                        print(f"LocationAgent: Error searching for '{location_name}': {e}")
                        continue
        print(f"LocationAgent: Extraction complete. Total locations extracted: {len(locations)}")
        return locations

    def enrich_location_data(self, location: Location) -> Location:
        print(f"LocationAgent: Enriching location data for '{location.name}' (lat, lng) = ({location.lat}, {location.lng})")
        try:
            places_result = self.gmaps.places(location.name, location=f"{location.lat},{location.lng}", radius=1000)
            if places_result.get('results'):
                place = places_result['results'][0]
                place_details = self.gmaps.place(place['place_id'], fields=['name', 'photos', 'formatted_address', 'rating'])
                if 'photos' in place_details['result']:
                    photo_ref = place_details['result']['photos'][0]['photo_reference']
                    location.image = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={os.getenv('GOOGLE_MAPS_API_KEY')}"
                    print(f"LocationAgent: Photo fetched for '{location.name}'.")
            else:
                print(f"LocationAgent: No photo found for '{location.name}'.")
        except Exception as e:
            print(f"LocationAgent: Error enriching '{location.name}': {e}")
        return location

    def process_itinerary(self, itinerary: str) -> List[Dict]:
        print("LocationAgent: Starting to parse itinerary text...")
        locations = self.extract_locations_from_itinerary(itinerary)
        enriched_locations = []
        for loc in locations:
            enriched_loc = self.enrich_location_data(loc)
            enriched_locations.append({
                'name': enriched_loc.name,
                'description': enriched_loc.description,
                'lat': enriched_loc.lat,
                'lng': enriched_loc.lng,
                'image': enriched_loc.image
            })
        print(f"LocationAgent: Finished parsing and enriching locations. Returning {len(enriched_locations)} locations.")
        return enriched_locations 