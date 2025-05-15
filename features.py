import json
from dateutil import parser
from fuzzywuzzy import process

INPUT_FILE = "cleaned_appraisals_dataset.json"
OUTPUT_FILE = "feature_engineered_appraisals_dataset.json"

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

    # 1. Manual check first
    if val in manual_type_map:
        return manual_type_map[val]

    # 2. Fuzzy fallback to catch close things like "semi detached"
    match, score = process.extractOne(val, CANONICAL_TYPES, scorer=process.fuzz.partial_ratio)
    return match if score >= 80 else None


def same_property_type(appraisal):
    subject = appraisal['subject']
    subject_raw = subject.get('structure_type')
    subject_type = map_to_property_type(subject_raw)

    if not subject_type:
        return appraisal  # skip if undefined/none

    for comp in appraisal['comps']:
        comp_raw = comp.get('prop_type')
        comp_type = map_to_property_type(comp_raw)
        comp['same_property_type'] = int(subject_type == comp_type)

    for property in appraisal['properties']:
        property_raw = property.get('property_sub_type')
        property_type = map_to_property_type(property_raw)
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

    if not subject_lot_size:
        return appraisal
    
    for comp in appraisal['comps']:
        comp_lot_size = comp.get('lot_size_sf')
        if comp_lot_size:
            comp['lot_size_diff_sf'] = subject_lot_size - comp_lot_size
        else:
            comp['lot_size_diff_sf'] = None

    for property in appraisal['properties']:
        property_lot_size = property.get('lot_size_sf')
        if property_lot_size:
            property['lot_size_diff_sf'] = subject_lot_size - property_lot_size
        else:
            property['lot_size_diff_sf'] = None

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
            comp['total_rooms_diff'] = None

    for property in appraisal['properties']:
        property_rooms = property.get('room_count')
        if property_rooms:
            property['total_rooms_diff'] = subject_rooms - comp_rooms
        else:
            property['total_rooms_diff'] = None
    
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
        room_diff(appraisal)


        feature_engineered.append(appraisal)


    with open(OUTPUT_FILE, "w") as f:
        json.dump({"appraisals": feature_engineered}, f, indent=2)

    print(f"Saved cleaned JSON to {OUTPUT_FILE}")
    

if __name__ == "__main__":
    add_new_features()    

