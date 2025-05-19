# import os
# import json
# import time
# from tqdm import tqdm
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut
# from openai import OpenAI

# # --- CONFIG ---
# INPUT_FILE = "feature_engineered_appraisals_dataset.json"
# CACHE_FILE = "geocode_address.json"  
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

# # --- LLM SYSTEM PROMPT ---
# SYSTEM_PROMPT = (
#     "You are a geocoding assistant trained to clean and standardize Canadian mailing addresses strictly "
#     "for geolocation purposes.\n\n"
#     "Your job is to rewrite any input into this format exactly:\n"
#     "[unit-civic number] [Street Name Capitalized], [City Capitalized], [Province Abbreviation] [Postal Code], Canada\n\n"
#     "Rules:\n"
#     "- Use commas between address parts (street, city, province, postal code, country)\n"
#     "- Ensure proper capitalization (e.g., 'Kemptville', 'ON')\n"
#     "- Postal codes must have a space between the 3rd and 4th character (e.g., 'T2N 3B8')\n"
#     "- If the address includes a unit/civic format (e.g., '119 110'), rewrite it as '110-119'\n"
#     "- Do not include neighborhood names, regions, or repetitions — just the precise mailing address\n"
#     "- Your response must only include the final cleaned address, with no explanation or extra text"
# )

# # --- Functions ---
# def normalize_for_geocoding(address):
#     if not address:
#         return None
#     return address.lower().strip().replace("drive", "dr").replace("road", "rd").replace("street", "st").replace("avenue", "ave")

# def enrich_address(address, context_parts):
#     base = normalize_for_geocoding(address)
#     if not base:
#         return None
#     components = [base]
#     for part in context_parts:
#         if part:
#             part_str = str(part).strip()
#             if part_str.lower() not in base:
#                 components.append(part_str)
#     return ", ".join(components)

# def safe_geocode(geolocator, address):
#     try:
#         return geolocator.geocode(address, timeout=10)
#     except GeocoderTimedOut:
#         time.sleep(2)
#         return safe_geocode(geolocator, address)
#     except Exception as e:
#         print(f"Error geocoding '{address}': {e}")
#         return None

# def clean_address_with_gpt(raw_address):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": f"Please clean and standardize this address: {raw_address}"}
#             ],
#             temperature=0
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"GPT error for '{raw_address}': {e}")
#         return None

# # --- Load Files ---
# with open(INPUT_FILE, "r") as f:
#     data = json.load(f)

# if os.path.exists(CACHE_FILE):
#     with open(CACHE_FILE, "r") as f:
#         geocoded = json.load(f)
# else:
#     geocoded = {}

# # --- Collect Unique Addresses ---
# unique_addresses = {}
# for appraisal in data["appraisals"]:
#     subject = appraisal.get("subject", {})
#     addr = normalize_for_geocoding(subject.get("address"))
#     enriched = enrich_address(addr, [
#         subject.get("subject_city_province_zip"),
#         subject.get("municipality_district")
#     ])
#     if addr and enriched:
#         unique_addresses[addr] = enriched

#     for comp in appraisal.get("comps", []):
#         addr = normalize_for_geocoding(comp.get("address"))
#         enriched = enrich_address(addr, [comp.get("city_province")])
#         if addr and enriched:
#             unique_addresses[addr] = enriched

#     for prop in appraisal.get("properties", []):
#         addr = normalize_for_geocoding(prop.get("address"))
#         enriched = enrich_address(addr, [
#             prop.get("city"),
#             prop.get("province"),
#             prop.get("postal_code")
#         ])
#         if addr and enriched:
#             unique_addresses[addr] = enriched

# print(f"📍 Unique normalized addresses to check: {len(unique_addresses)}")

# # --- Geocode Process ---
# geolocator = Nominatim(user_agent="comp-geocoder")

# for norm_addr, enriched_addr in tqdm(unique_addresses.items()):
#     if norm_addr in geocoded and geocoded[norm_addr] is not None:
#         continue  # Already successfully geocoded

#     print(f"🔎 Attempting: {enriched_addr}")
#     location = safe_geocode(geolocator, enriched_addr)
#     if location:
#         geocoded[norm_addr] = {
#             "lat": location.latitude,
#             "lon": location.longitude,
#             "source": "nominatim",
#             "parsed": enriched_addr
#         }
#     else:
#         print(f"❌ Nominatim failed. Trying GPT for: {enriched_addr}")
#         gpt_address = clean_address_with_gpt(enriched_addr)
#         if gpt_address:
#             location = safe_geocode(geolocator, gpt_address)
#             if location:
#                 print(f"✅ GPT success: {gpt_address}")
#                 geocoded[norm_addr] = {
#                     "lat": location.latitude,
#                     "lon": location.longitude,
#                     "source": "gpt",
#                     "parsed": gpt_address
#                 }
#             else:
#                 print(f"❌ GPT failed geocode for: {gpt_address}")
#                 geocoded[norm_addr] = None
#         else:
#             print(f"❌ GPT parsing failed for: {enriched_addr}")
#             geocoded[norm_addr] = None

#     # Save incrementally
#     with open(CACHE_FILE, "w") as f:
#         json.dump(geocoded, f, indent=2)

#     time.sleep(1)

# print(f"✅ Final cache saved to {CACHE_FILE}")

import os
import json
import time
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from openai import OpenAI

# Config 
CACHE_FILE = "geocoded_addresses.json"
MISSING_FILE = "missing_addresses.txt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Helper functions
def normalize_address(address):
    return address.lower().strip()

def safe_geocode(geolocator, address):
    try:
        return geolocator.geocode(address, timeout=10)
    except GeocoderTimedOut:
        time.sleep(2)
        return safe_geocode(geolocator, address)
    except Exception as e:
        print(f"Geocode error for '{address}': {e}")
        return None

def clean_address_with_gpt(raw_address):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a geocoding assistant trained to clean and standardize Canadian mailing addresses strictly "
                    "for geolocation purposes.\n\n"
                    "Your job is to rewrite any input into this format exactly:\n"
                    "[unit-civic number] [Street Name Capitalized], [City Capitalized], [Province Abbreviation] [Postal Code], Canada\n\n"
                    "Rules:\n"
                    "- Use commas between address parts (street, city, province, postal code, country)\n"
                    "- Ensure proper capitalization (e.g., 'Kemptville', 'ON')\n"
                    "- Postal codes must have a space between the 3rd and 4th character (e.g., 'T2N 3B8')\n"
                    "- If the address includes a unit/civic format (e.g., '119 110'), rewrite it as '110-119'\n"
                    "- Do not include neighborhood names, regions, or repetitions — just the precise mailing address\n"
                    "- Your response must only include the final cleaned address, with no explanation or extra text"
                )},
                {"role": "user", "content": f"Please clean and standardize this address: {raw_address}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT error for '{raw_address}': {e}")
        return None

# Load cache 
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        geocoded = json.load(f)
else:
    geocoded = {}

# Load missing list 
with open(MISSING_FILE, "r") as f:
    missing_addresses = [normalize_address(line) for line in f if line.strip()]

# Main process 
geolocator = Nominatim(user_agent="comp-geocoder")
added = 0

for raw_address in tqdm(missing_addresses):
    if raw_address in geocoded and geocoded[raw_address] is not None:
        continue

    print(f"📍 Geocoding: {raw_address}")
    location = safe_geocode(geolocator, raw_address)
    if location:
        geocoded[raw_address] = {
            "lat": location.latitude,
            "lon": location.longitude,
        }
        added += 1
    else:
        print(f"⚠️ Nominatim failed. Trying GPT to clean: {raw_address}")
        cleaned = clean_address_with_gpt(raw_address)
        if cleaned:
            location = safe_geocode(geolocator, cleaned)
            if location:
                print(f"GPT cleaned success: {cleaned}")
                geocoded[raw_address] = {
                    "lat": location.latitude,
                    "lon": location.longitude,
                }
                added += 1
            else:
                print(f"GPT cleaned address failed to geocode: {cleaned}")
                geocoded[raw_address] = None
        else:
            print(f"GPT failed to parse: {raw_address}")
            geocoded[raw_address] = None

    # Save incrementally
    with open(CACHE_FILE, "w") as f:
        json.dump(geocoded, f, indent=2)

    time.sleep(1)

print(f"\nGeocoding complete — {added} new addresses added to {CACHE_FILE}")
