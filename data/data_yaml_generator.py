# Code retrieved from Edje Electronics's Tutorial on Working with YOLO Models,
# but slightly modified to meet our needs:
# https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models

# Python function to automatically create data.yaml config file
# 1. Reads "classes.txt" file to get list of class names
# 2. Creates data dictionary with correct paths to folders, number of classes, and names of classes
# 3. Writes data in YAML format to data.yaml

import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    # Read class.txt to get class names
    if not os.path.exists(path_to_classes_txt):
        print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
        return
    with open(path_to_classes_txt, 'r') as f:
        classes = []
        for line in f.readlines():
            if len(line.strip()) == 0: continue
            classes.append(line.strip())
        number_of_classes = len(classes)

    # Retrieve the directory of classes.txt and split on 'data' folder
    split_dir = os.path.abspath(path_to_classes_txt)
    split_parts = split_dir.split('data', 1)
    if len(split_parts) != 2:
        print(f"Error: 'data' not found in the path {split_dir}")
        return
    
    # Get the directory of the second part and append train/images and test/images
    base_path = split_parts[0] + 'data'
    relative_path = split_parts[1]
    train_path = os.path.join(os.path.dirname(relative_path), 'train/images')
    val_path = os.path.join(os.path.dirname(relative_path), 'test/images')
    
    # Create data dictionary
    data = {
        'path': base_path,
        'train': train_path,
        'val': val_path,
        'nc': number_of_classes,
        'names': classes
    }

    # Write data to YAML file
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'Created config file at {path_to_data_yaml}')