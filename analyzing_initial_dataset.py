import json

INPUT_FILE = "appraisals_dataset.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

EXTRACTED_DATA_FILE= "gpt_extracted_features_appraisals.json"

with open(EXTRACTED_DATA_FILE, "r") as f:
            extracted_data = json.load(f)


unique_subject_conditions = []
unique_comp_conditions = []
unique_property_conditions = []

def clean_extracted_conditions(appraisal):
    subject = appraisal['subject']
    subject_cond = subject.get('condition')  

    if subject_cond.lower().strip() not in unique_subject_conditions:
        unique_subject_conditions.append(subject_cond.lower().strip())

    for comp in appraisal['comps']:
        comp_cond = comp.get('condition')
        if comp_cond.lower().strip() not in unique_comp_conditions:
            unique_comp_conditions.append(comp_cond.lower().strip())

    for prop in appraisal['properties']:
        prop_cond = prop.get('condition')
        
        if prop_cond: 
            normalized = prop_cond.strip().lower()
            if normalized and normalized not in unique_property_conditions:
                unique_property_conditions.append(normalized)

total_appraisals = len(data['appraisals'])

total_extracted_appraisals = len(extracted_data['appraisals'])

print(f"Total initial appraisals: {total_appraisals}")
print(f"Total extracted appraisals: {total_extracted_appraisals}")

total_properties = 0

for appraisal in data['appraisals']:
    total_properties+=len(appraisal['properties'])

total_extracted_properties = 0

for appraisal in extracted_data['appraisals']:
    clean_extracted_conditions(appraisal)
    total_extracted_properties+=len(appraisal['properties'])

print(f"Total initial properties: {total_properties}")

print(f"Total extracted properties: {total_extracted_properties}")

print(unique_subject_conditions)
print(unique_comp_conditions)
print(unique_property_conditions)


# average_properties_per_appraisal = total_properties / total_appraisals

# print(f"Average properties per appraisal: {average_properties_per_appraisal}")

# for property in data['appraisals'][0]['properties'][:10]:
#     print(property['public_remarks'] + "\n")

