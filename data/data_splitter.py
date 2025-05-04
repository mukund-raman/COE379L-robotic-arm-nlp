import shutil
import os
from pathlib import Path
import sys
import random

def initialize_train_test_dirs(data_dir):
    # Remove existing train/test directories if necessary
    try:
        shutil.rmtree(f'{data_dir}-split/train')
        shutil.rmtree(f'{data_dir}-split/test')
    except:
        pass

    # Create train directories for images
    Path(f'{data_dir}-split/train/images').mkdir(parents=True, exist_ok=True)
    Path(f'{data_dir}-split/train/labels').mkdir(parents=True, exist_ok=True)

    # Create test directories for images
    Path(f'{data_dir}-split/test').mkdir(parents=True, exist_ok=True)
    Path(f'{data_dir}-split/test/images').mkdir(parents=True, exist_ok=True)
    Path(f'{data_dir}-split/test/labels').mkdir(parents=True, exist_ok=True)

def split_data(data_dir, train_ratio=0.8):
    # Initialize train/test directories
    initialize_train_test_dirs(data_dir)
    
    # Copy over the classes text file to the data directory
    shutil.copyfile(os.path.join(data_dir, 'classes.txt'), os.path.join(f'{data_dir}-split', 'classes.txt'))
    
    # Define path to input dataset
    input_image_path = os.path.join(data_dir, 'images')
    input_label_path = os.path.join(data_dir, 'labels')
    
    # Define paths to image and annotation folders
    train_img_path = f'{data_dir}-split/train/images'
    train_txt_path = f'{data_dir}-split/train/labels'
    test_img_path = f'{data_dir}-split/test/images'
    test_txt_path = f'{data_dir}-split/test/labels'
    
    # Get list of all images and annotation files
    img_file_list = [path for path in Path(input_image_path).rglob('*')]
    txt_file_list = [path for path in Path(input_label_path).rglob('*')]
    print(f'Number of image files: {len(img_file_list)}')
    print(f'Number of annotation files: {len(txt_file_list)}')
    
    # Determine number of files to move to each folder
    file_num = len(img_file_list)
    train_percent = 0.80
    train_num = int(file_num*train_percent)
    test_num = file_num - train_num
    print('Images moving to train: %d' % train_num)
    print('Images moving to test: %d' % test_num)
    
    # Select files randomly and more to train to test
    for i, set_num in enumerate([train_num, test_num]):
        if i == 0: # Copy first set of files to train folders
            new_img_path, new_txt_path = train_img_path, train_txt_path
        elif i == 1: # Copy second set of files to the validation folders
            new_img_path, new_txt_path = test_img_path, test_txt_path
        
        for ii in range(set_num):
            # Randomly select an image file from the list
            img_path = random.choice(img_file_list)
            img_fn = img_path.name
            base_fn = img_path.stem
            txt_fn = base_fn + '.txt'
            txt_path = os.path.join(input_label_path, txt_fn)

            # Copy image and annotation files to the new directories
            shutil.copy(img_path, os.path.join(new_img_path, img_fn))
            if os.path.exists(txt_path):
                shutil.copy(txt_path, os.path.join(new_txt_path, txt_fn))

            img_file_list.remove(img_path)