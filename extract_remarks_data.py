import json
import os
import re
from tqdm import tqdm
from openai import OpenAI

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=api_key)

INPUT_FILE = "appraisals_dataset.json"
CACHE_FILE = "gpt_extracted_features_appraisals.json"

def build_prompt(remarks: str) -> str:
    return f"""
You are a real estate data extractor. From the following listing description ("public_remarks"), extract structured property data into a JSON object.

If a value is missing or ambiguous, return null. Do not guess. Use the public_remarks as the single source of truth.

Extract the following fields:

- num_beds: The total number of bedrooms (integer)
- num_full_baths: The total number of full bathrooms (integer)
- num_half_baths: The total number of half bathrooms (integer)
- bath_score: Total combines number of bathrooms (float, computed as full + 0.5 * half)
- lot size (float, converted to acres if not already in acres)
- GLA (float, converted to square feet if not already in square feet)
- has_garage (1/0/null)
- has_basement (1/0/null)
- basement_finished (1/0/null)
- basement_walkout (1/0/null)
- basement_full_part (1/0.5/null)
- has_deck (1/0/null)
- has_fireplace (1/0/null)
- central_air (1/0/null)
- window_unit (1/0/null)
- heating_type (Forced Air/Baseboard/null, string)
- has_pool (1/0/null)
- has_renovation (1/0/null)
- flooring_types (array of strings)
- style (string)
- condition (Excellent, Good, Average, Fair) YOU MAY ONLY USE ONE OF THESE 4 CONDITION OPTIONS, DO NOT GET CREATIVE

Return JSON with exactly these keys. If unsure, return nulls.
Do not include any code block formatting or explanatory text.

Public Remarks:
\"\"\"{remarks.strip()}\"\"\"
"""

def extract_features_from_remarks(remarks):
    if not remarks:
        return {}
    prompt = build_prompt(remarks)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You extract property features from real estate descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception as e:
        print("GPT extraction failed:", e)
        return {}

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cached_data = json.load(f)
            cleaned_appraisals = cached_data.get("appraisals", [])
            cached_addresses = {
                prop.get('address')
                for appraisal in cleaned_appraisals
                for prop in appraisal.get("properties", [])
                if prop.get('address')
            }
    else:
        cleaned_appraisals = []
        cached_addresses = set()

    with tqdm(total=len(data["appraisals"]), desc="Extracting from Remarks") as pbar:
        for appraisal in data["appraisals"]:
            order_id = appraisal.get("orderID")
            enhanced_properties = []

            for prop in appraisal.get("properties", []):
                prop_address = prop.get('address')
                if not prop_address or prop_address in cached_addresses:
                    continue

                remarks = prop.get("public_remarks", "")
                extracted = extract_features_from_remarks(remarks)
                prop.update(extracted)
                enhanced_properties.append(prop)
                cached_addresses.add(prop_address)

            # Only append if there are processed properties
            if enhanced_properties:
                cleaned_appraisals.append({
                    "orderID": order_id,
                    "subject": appraisal.get("subject"),
                    "comps": appraisal.get("comps", []),
                    "properties": enhanced_properties
                })

                # Save incrementally
                with open(CACHE_FILE, "w") as f:
                    json.dump({"appraisals": cleaned_appraisals}, f, indent=2)

            pbar.update(1)

    print(f"\nExtracted dataset saved to {CACHE_FILE}")

if __name__ == "__main__":
    main()
