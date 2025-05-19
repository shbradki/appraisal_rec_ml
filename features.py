import json
from dateutil import parser
from fuzzywuzzy import process
from geopy.distance import geodesic

# Config
INPUT_FILE = "cleaned_appraisals_dataset.json"
OUTPUT_FILE = "feature_engineered_appraisals_dataset.json"

ADDRESS_FILE = "geocoded_addresses.json"

with open(ADDRESS_FILE, "r") as f:
            address_data = json.load(f)

CANONICAL_TYPES = [
    "Townhouse", "Detached", "Condominium", "Semi Detached",
    "High Rise Apartment", "Low Rise Apartment", "Duplex", "Triplex", "Fourplex"
]

manual_type_map = {
    "rural resid": "Detached",
    "rural residential": "Detached",
    "single family": "Detached",
    "single family residence": "Detached",
    "overunder": "Duplex",
    "4 plex": "Fourplex",
    "triplex": "Triplex",
    "duplex": "Duplex",
    "over under": "Duplex",
    "condo apt": "Condominium",
    "condo apartment": "Condominium",
    "condo/apt unit": "Condominium",
    "common element condo": "Condominium",
    "row unit": "Townhouse",
    "row unit 2 storey": "Townhouse",
    "row unit 3 storey": "Townhouse",
    "stacked": "Townhouse",
    "mobiletrailer": "Detached",
    "mobile home": "Detached",
    "mobile": "Detached",
    "link": "Semi Detached",
    "farm": "Detached",
    "vacant land": None,
    "residential land": None,
    "residential": "",
    "locker": None,
    "other": None,
    "": None,
    None: None
}


def sold_recently(appraisal):
    subject = appraisal['subject']
    subject_effective_data = parser.parse(subject['effective_date'])

    for comp in appraisal['comps']:
        sale_date = parser.parse(comp.get('sale_date'))
        days_ago_sold = (subject_effective_data-sale_date).days
        if days_ago_sold <= 90:
            comp['sold_recently'] = 1
        else:
            comp['sold_recently'] = 0
        

    for property in appraisal['properties']:
        close_date = parser.parse(property.get('close_date'))
        days_ago_sold = (subject_effective_data-close_date).days
        if days_ago_sold <= 90:
            property['sold_recently'] = 1
        else:
            property['sold_recently'] = 0

    return appraisal

def map_to_property_type(raw):
    if not raw:
        return None

    val = str(raw).lower().strip().replace(",", "").replace("-", " ")

    # Manual check first
    if val in manual_type_map:
        return manual_type_map[val]

    # Fuzzy fallback to catch close things
    match, score = process.extractOne(val, CANONICAL_TYPES, scorer=process.fuzz.partial_ratio)
    return match if score >= 80 else None


def same_property_type(appraisal):
    subject = appraisal['subject']
    subject_raw = subject.get('structure_type')
    subject_type = map_to_property_type(subject_raw)
    subject['property_type'] = subject_type

    if not subject_type:
        return appraisal  

    for comp in appraisal['comps']:
        comp_raw = comp.get('prop_type')
        comp_type = map_to_property_type(comp_raw)
        comp['property_type'] = comp_type
        comp['same_property_type'] = int(subject_type == comp_type)

    for property in appraisal['properties']:
        property_raw = property.get('property_sub_type')
        property_type = map_to_property_type(property_raw)
        property['property_type'] = property_type
        property['same_property_type'] = int(subject_type == property_type)
        

    return appraisal

def effective_age_diff(appraisal):
    subject = appraisal['subject']
    subject_effective_age = subject.get("effective_age")
    
    if not subject_effective_age:
        return appraisal

    for comp in appraisal['comps']:
        comp_age = comp.get('age')
        if comp_age:
            comp['effective_age_diff'] = subject_effective_age-comp_age
        else:
            comp['effective_age_diff'] = None

    for property in appraisal['properties']:
        property_age = property.get('age')
        if property_age:
            property['effective_age_diff'] = subject_effective_age-property_age
        else:
            property['effective_age_diff'] = None

    return appraisal

def subject_age_diff(appraisal):
    subject = appraisal['subject']
    subject_age = subject.get("subject_age")

    if not subject_age:
        return appraisal
    
    for comp in appraisal['comps']:
        comp_age = comp.get('age')
        if comp_age:
            comp['subject_age_diff'] = subject_age-comp_age
        else:
            comp['subject_age_diff'] = None

    for property in appraisal['properties']:
        property_age = property.get('age')
        if property_age:
            property['subject_age_diff'] = subject_age-property_age
        else:
            property['subject_age_diff'] = None

    return appraisal

def lot_size_diff(appraisal):
    subject = appraisal['subject']
    subject_lot_size = subject.get('lot_size_sf')

    for comp in appraisal['comps']:
        comp_lot_size = comp.get('lot_size_sf')
        if subject_lot_size is not None and comp_lot_size is not None:
            comp['lot_size_diff_sf'] = subject_lot_size - comp_lot_size
        else:
            comp['lot_size_diff_sf'] = None

    for property in appraisal['properties']:
        property_lot_size = property.get('lot_size_sf')
        if subject_lot_size is not None and property_lot_size is not None:
            property['lot_size_diff_sf'] = subject_lot_size - property_lot_size
        else:
            property['lot_size_diff_sf'] = None

    return appraisal

def gla_diff(appraisal):
    subject = appraisal['subject']
    subject_gla = subject.get('gla')

    if not subject_gla:
        return appraisal
    
    for comp in appraisal['comps']:
        comp_gla = comp.get('gla')
        if comp_gla:
            comp['gla_diff'] = subject_gla - comp_gla
        else:
            comp['gla_diff'] = None

    for property in appraisal['properties']:
        property_gla = property.get('gla')
        if property_gla:
            property['gla_diff'] = subject_gla - property_gla
        else:
            property['gla_diff'] = None

    return appraisal

def room_diff(appraisal):
    subject = appraisal['subject']
    subject_rooms = subject.get('room_count')

    if not subject_rooms:
        return appraisal

    for comp in appraisal['comps']:
        comp_rooms = comp.get('room_count')
        if comp_rooms:
            comp['room_count_diff'] = subject_rooms - comp_rooms
        else:
            comp['room_count_diff'] = None

    for property in appraisal['properties']:
        property_rooms = property.get('room_count')
        if property_rooms:
            property['room_count_diff'] = subject_rooms - property_rooms
        else:
            property['room_count_diff'] = None
    
    return appraisal

def bedroom_diff(appraisal):
    subject = appraisal['subject']
    subject_bedrooms = subject.get('num_beds')

    if not subject_bedrooms:
        return appraisal

    for comp in appraisal['comps']:
        comp_bedrooms = comp.get('num_beds')
        if comp_bedrooms:
            comp['bedrooms_diff'] = subject_bedrooms - comp_bedrooms
        else:
            comp['bedrooms_diff'] = None

    for property in appraisal['properties']:
        property_bedrooms = property.get('num_beds')
        if property_bedrooms:
            property['bedrooms_diff'] = subject_bedrooms - property_bedrooms
        else:
            property['bedrooms_diff'] = None
    
    return appraisal

def bath_score_diff(appraisal):
    subject = appraisal['subject']
    subject_bath_score = subject.get('bath_score')

    if not subject_bath_score:
        return appraisal

    for comp in appraisal['comps']:
        comp_bath_score = comp.get('bath_score')
        if comp_bath_score:
            comp['bath_score_diff'] = subject_bath_score - comp_bath_score
        else:
            comp['bath_score_diff'] = None

    for property in appraisal['properties']:
        property_bath_score = property.get('bath_score')
        if property_bath_score:
            property['bath_score_diff'] = subject_bath_score - property_bath_score
        else:
            property['bath_score_diff'] = None
    
    return appraisal

def full_bath_diff(appraisal):
    subject = appraisal['subject']
    subject_fulls = subject.get('num_full_baths')

    if not subject_fulls:
        return appraisal

    for comp in appraisal['comps']:
        comp_fulls = comp.get('num_full_baths')
        if comp_fulls:
            comp['full_baths_diff'] = subject_fulls - comp_fulls
        else:
            comp['full_baths_diff'] = None

    for property in appraisal['properties']:
        property_fulls = property.get('num_full_baths')
        if property_fulls:
            property['full_baths_diff'] = subject_fulls - property_fulls
        else:
            property['full_baths_diff'] = None
    
    return appraisal

def half_bath_diff(appraisal):
    subject = appraisal['subject']
    subject_halfs = subject.get('num_half_baths')

    if not subject_halfs:
        return appraisal

    for comp in appraisal['comps']:
        comp_halfs = comp.get('num_half_baths')
        if comp_halfs:
            comp['half_baths_diff'] = subject_halfs - comp_halfs
        else:
            comp['half_baths_diff'] = None

    for property in appraisal['properties']:
        property_halfs = property.get('num_half_baths')
        if property_halfs:
            property['half_baths_diff'] = subject_halfs - property_halfs
        else:
            property['half_baths_diff'] = None
    
    return appraisal

def add_geocoded_addresses(appraisal):
    def get_lat_lon(address):
        data = address_data.get(address)
        if data and isinstance(data, dict):
            return data.get('lat'), data.get('lon')
        return None, None

    subject = appraisal['subject']
    subject_address = subject.get('address').lower()
    subject['lat'], subject['lon'] = get_lat_lon(subject_address)

    for comp in appraisal.get('comps', []):
        comp_address = comp.get('address').lower()
        comp['lat'], comp['lon'] = get_lat_lon(comp_address)

    for prop in appraisal.get('properties', []):
        prop_address = prop.get('address').lower()
        prop['lat'], prop['lon'] = get_lat_lon(prop_address)

    return appraisal

def get_distance_to_subject(appraisal):

    def get_dist(sub_lat, sub_lon, lat, lon):
        try:
            dist_km = geodesic(
                (sub_lat, sub_lon), (lat, lon)
            ).km
            return round(dist_km, 3)
        except Exception as e:
            print(f"Distance error for {comp_address}: {e}")
            return None

    subject = appraisal['subject']
    subject_lat = subject.get('lat')
    subject_lon = subject.get('lon')

    if subject_lat is None or subject_lon is None:
        print(subject.get('address'))
        return appraisal 

    for comp in appraisal['comps']:

        # Skip if already has a valid distance
        if comp.get('distance_to_subject_km') is not None:
            continue

        comp_address = comp.get('address')
        if not comp_address:
            continue

        cached = address_data.get(comp_address.lower())
        if cached and isinstance(cached, dict):
            comp_lat = cached.get('lat')
            comp_lon = cached.get('lon')
            if comp_lat is not None and comp_lon is not None:

                # Calculate geodesic distance in kilometers
                comp['distance_to_subject_km'] = get_dist(subject_lat, subject_lon, comp_lat, comp_lon)

    for property in appraisal['properties']:
        property_address = property.get('address')
        if not property_address: 
            continue

        cached = address_data.get(property_address.lower())
        if cached and isinstance(cached, dict):
            property_lat = cached.get('lat')
            property_lon = cached.get('lon')
            if property_lat is not None and property_lon is not None:

                # Calculate geodesic distance in kilometers
                property['distance_to_subject_km'] = get_dist(subject_lat, subject_lon, property_lat, property_lon)
        
    return appraisal 
        

def add_new_features():
    with open(INPUT_FILE, "r") as f:
            data = json.load(f)

    feature_engineered = []
    
    for appraisal in data["appraisals"]:
        
        sold_recently(appraisal)
        same_property_type(appraisal)
        effective_age_diff(appraisal)
        subject_age_diff(appraisal)
        lot_size_diff(appraisal)
        gla_diff(appraisal)
        room_diff(appraisal)
        bedroom_diff(appraisal)
        bath_score_diff(appraisal)
        full_bath_diff(appraisal)
        half_bath_diff(appraisal)

        add_geocoded_addresses(appraisal)
        get_distance_to_subject(appraisal)

        feature_engineered.append(appraisal)


    with open(OUTPUT_FILE, "w") as f:
        json.dump({"appraisals": feature_engineered}, f, indent=2)

    print(f"Saved cleaned JSON to {OUTPUT_FILE}")
    

if __name__ == "__main__":
    add_new_features()    

