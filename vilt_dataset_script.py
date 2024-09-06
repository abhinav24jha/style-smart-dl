import json
import random
import os

RAW_DATASET_JSON_FILES_PATH = "raw_dataset_multimode_transformers"
FASHION_DISASTER_JSON_FILE_PATH = "fashion_disasters_images.json"

def load_json(file_path):
    # loads the content at file_path from a json file and returns the content

    with open(file_path, 'r') as f:
        return json.load(f)
    

def find_candidates(data, condition_func):
    # Returns all candidates that match a given condition
    return [d for d in data if condition_func(d)]

def generate_contrastive_dataset(raw_data_json_files,
                                 fashion_disaster_json_file,
                                 max_combinations_per_anchor):
    
    # this essentially combines all the raw_dataset .json files into one file for easier access

    raw_data = []
    for file in raw_data_json_files:
        data = load_json(file)
        raw_data.extend(data['dataset'])

    # this loads in the fashion_disasters_images.json
    fashion_disaster_data = load_json(fashion_disaster_json_file)

    #shuffle data for initial randomness
    random.shuffle(raw_data)
    random.shuffle(fashion_disaster_data)

    #creating a list to store the new generated datset
    generated_dataset = []

    for anchor in raw_data:

        # anchor = random.choice(raw_data)
        used_images = {anchor['image']}
        
        # Find all valid positives
        positives = find_candidates(raw_data, lambda d: d['weather'] == anchor['weather'] and d['schedule'] == anchor['schedule'] and d['image'] != anchor['image'])

        # Find all valid contextual negatives
        contextual_negatives = find_candidates(raw_data, lambda d: (d['weather'] != anchor['weather'] or d['schedule'] != anchor['schedule']) and d['image'] not in used_images)

        # Find all valid fashion negatives
        fashion_negatives = find_candidates(fashion_disaster_data, lambda d: d['weather'] == anchor['weather'] and d['schedule'] == anchor['schedule'])

        # Randomly sample up to max_combinations_per_anchor from each set of candidates
        num_combinations = min(len(positives), len(contextual_negatives), len(fashion_negatives), max_combinations_per_anchor)


        for _ in range(num_combinations):
            positive = random.choice(positives)
            contextual_negative = random.choice(contextual_negatives)
            fashion_negative = random.choice(fashion_negatives)

            datapoint = {
                "anchor": anchor,
                "positive": positive,
                "contextual_negative": contextual_negative,
                "fashion_negative": fashion_negative
            }
            generated_dataset.append(datapoint)
    
    return generated_dataset

def save_dataset(dataset, output_file):
    # saving the generated_dataset in a json file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    
def save_quality_control_subset(dataset, output_file, sample_size):
    # saves a subset of the json file of generated_dataset, for manual_inspection
    subset = random.sample(dataset, min(sample_size, len(dataset)))
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=4)

def main():
    # path to the raw_dataset .json files
    total_raw_data_json_files = 5
    raw_data_files = [os.path.join(RAW_DATASET_JSON_FILES_PATH, f"raw_dataset_{i}.json") for i in range(1, total_raw_data_json_files + 1)]

    # generating the dataset
    generated_dataset = generate_contrastive_dataset(raw_data_json_files=raw_data_files,
                                                     fashion_disaster_json_file=FASHION_DISASTER_JSON_FILE_PATH,
                                                     max_combinations_per_anchor=3)
    
    # saving this generated dataset into a json file
    save_dataset(generated_dataset, 'generated_dataset.json')

    # # saving the manual control quality check
    # save_quality_control_subset(dataset=generated_dataset, output_file='quality_control_subset.json', sample_size=10)

if __name__ == "__main__":
    main()