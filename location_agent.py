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
        """Extract location names from the itinerary text."""
        # Split into days and extract locations
        days = re.split(r'Day \d+:', itinerary)
        locations = []
        seen_locations = set()  # Track unique locations
        
        for day in days:
            # Look for location patterns like "Visit X", "Explore X", "X is a", etc.
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
                    
                    # Skip if we've already processed this location
                    if location_name.lower() in seen_locations:
                        continue
                    
                    # Get the sentence containing this location
                    sentence_start = max(0, day.find(location_name) - 100)
                    sentence_end = min(len(day), day.find(location_name) + 100)
                    description = day[sentence_start:sentence_end].strip()
                    
                    # Clean up the location name
                    location_name = re.sub(r'^(the|a|an)\s+', '', location_name, flags=re.IGNORECASE)
                    location_name = location_name.strip()
                    
                    # Skip if location name is too short or too generic
                    if len(location_name) < 3 or location_name.lower() in ['city', 'town', 'place', 'area', 'region']:
                        continue
                    
                    # Get coordinates for the location
                    try:
                        # Add the destination city to improve geocoding accuracy
                        destination = os.getenv('DESTINATION', '')
                        search_query = f"{location_name}, {destination}" if destination else location_name
                        
                        # Geocode the location
                        geocode_result = self.gmaps.geocode(search_query)
                        
                        if geocode_result:
                            # Verify the result is in the correct city
                            formatted_address = geocode_result[0]['formatted_address']
                            if destination and destination.lower() not in formatted_address.lower():
                                print(f"Skipping {location_name} - not in {destination}")
                                continue
                                
                            location = Location(
                                name=location_name,
                                description=description,
                                lat=geocode_result[0]['geometry']['location']['lat'],
                                lng=geocode_result[0]['geometry']['location']['lng']
                            )
                            locations.append(location)
                            seen_locations.add(location_name.lower())
                            
                    except Exception as e:
                        print(f"Error geocoding {location_name}: {str(e)}")
                        continue

        return locations

    def enrich_location_data(self, location: Location) -> Location:
        """Enrich location data with additional information."""
        try:
            # Search for the location using Places API
            places_result = self.gmaps.places(
                location.name,
                location=f"{location.lat},{location.lng}",
                radius=1000
            )

            if places_result.get('results'):
                place = places_result['results'][0]
                
                # Get place details
                place_details = self.gmaps.place(
                    place['place_id'],
                    fields=['name', 'photos', 'formatted_address', 'rating']
                )

                # Update location with additional data
                if 'photos' in place_details['result']:
                    photo_ref = place_details['result']['photos'][0]['photo_reference']
                    location.image = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={os.getenv('GOOGLE_MAPS_API_KEY')}"

        except Exception as e:
            print(f"Error enriching location data for {location.name}: {str(e)}")

        return location

    def process_itinerary(self, itinerary: str) -> List[Dict]:
        """Process the itinerary and return enriched location data."""
        # Extract locations
        locations = self.extract_locations_from_itinerary(itinerary)
        
        # Enrich location data
        enriched_locations = []
        for location in locations:
            enriched_location = self.enrich_location_data(location)
            enriched_locations.append({
                'name': enriched_location.name,
                'description': enriched_location.description,
                'lat': enriched_location.lat,
                'lng': enriched_location.lng,
                'image': enriched_location.image
            })

        return enriched_locations 