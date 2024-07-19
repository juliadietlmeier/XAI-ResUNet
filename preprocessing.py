import os
import nibabel as nib
import numpy as np
import cv2
import csv
import random
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

output_base_path = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data'
train_csv_path = os.path.join(output_base_path, 'train.csv')
validation_csv_path = os.path.join(output_base_path, 'validation.csv')
test_csv_path = os.path.join(output_base_path, 'test.csv')

def setup_preprocess_logging(name, log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{name}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )

# Normalization function
def normalize_slice(img_slice):
    min_val = np.min(img_slice)
    max_val = np.max(img_slice)
    if max_val != min_val:
        img_slice = ((img_slice - min_val) / (max_val - min_val)) * 255
    else:
        img_slice = img_slice * 255
    return img_slice

# Function to normalize and save MRI and mask slices
def normalize_and_save_slices(mri_slice, mask_slice, output_path, slice_index, plane, center, patient, target_size=(256, 256)):
    mri_slice = normalize_slice(mri_slice).astype(np.uint8)
    mask_slice = normalize_slice(mask_slice).astype(np.uint8)

    mri_slice = cv2.resize(mri_slice, target_size, interpolation=cv2.INTER_AREA)
    mask_slice = cv2.resize(mask_slice, target_size, interpolation=cv2.INTER_NEAREST)

    mri_file_name = f"{center}_{patient}_{plane}_mri_slice_{str(slice_index).zfill(4)}.png"
    mask_file_name = f"{center}_{patient}_{plane}_mask_slice_{str(slice_index).zfill(4)}.png"

    Path(os.path.join(output_path, 'MRI')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_path, 'Mask')).mkdir(parents=True, exist_ok=True)

    cv2.imwrite(os.path.join(output_path, 'MRI', mri_file_name), mri_slice)
    cv2.imwrite(os.path.join(output_path, 'Mask', mask_file_name), mask_slice)

# Function to process MRI and mask slices for each plane
def process_slices(mri_img, mask_img, output_dir, center, patient, axis, plane):
    logging.info(f"Processing slices for {plane} plane, center: {center}, patient: {patient}")
    for slice_index in range(mri_img.shape[axis]):
        mri_slice = np.take(mri_img, slice_index, axis=axis)
        mask_slice = np.take(mask_img, slice_index, axis=axis)
        normalize_and_save_slices(mri_slice, mask_slice, output_dir, slice_index + 1, plane, center, patient, (512, 512))

# Function to process a single patient's data
def process_patient_data(mri_path, mask_path, output_dir, center, patient):
    try:
        mri_img = nib.load(mri_path).get_fdata()
        mask_img = nib.load(mask_path).get_fdata()
        process_slices(mri_img, mask_img, output_dir, center, patient, axis=0, plane='axial')
        process_slices(mri_img, mask_img, output_dir, center, patient, axis=1, plane='coronal')
        process_slices(mri_img, mask_img, output_dir, center, patient, axis=2, plane='sagittal')
        logging.info(f"Processed patient data for center: {center}, patient: {patient}")
    except Exception as e:
        logging.error(f"Error processing patient data for center: {center}, patient: {patient}, error: {e}")

# Function to process data for each center and patient
def process_center_and_patient(center_dir, patient_dir, base_dir, output_base_dir):
    patient_path = os.path.join(base_dir, center_dir, patient_dir)
    mri_path = os.path.join(patient_path, 'Preprocessed_Data', 'FLAIR_preprocessed.nii.gz')
    mask_path = os.path.join(patient_path, 'Masks', 'Consensus.nii.gz')

    if os.path.exists(mri_path) and os.path.exists(mask_path):
        output_dir = os.path.join(output_base_dir, center_dir, patient_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        process_patient_data(mri_path, mask_path, output_dir, center_dir, patient_dir)

# Main function to process all centers and patients using ProcessPoolExecutor
def process_all_centers_and_patients(base_dir, output_base_dir, max_workers=8):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")
    
    tasks = []
    for center_dir in os.listdir(base_dir):
        center_path = os.path.join(base_dir, center_dir)
        if os.path.isdir(center_path):
            for patient_dir in os.listdir(center_path):
                patient_path = os.path.join(center_path, patient_dir)
                if os.path.isdir(patient_path):
                    tasks.append((center_dir, patient_dir))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_center_and_patient, center_dir, patient_dir, base_dir, output_base_dir)
                   for center_dir, patient_dir in tasks]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing future: {e}")

# Function to find the unified bounding box across all slices
def find_unified_bounding_box(base_dirs, margin=10):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        for center_dir in os.listdir(base_dir):
            center_path = os.path.join(base_dir, center_dir)
            if not os.path.isdir(center_path):
                continue
            for patient_dir in os.listdir(center_path):
                patient_path = os.path.join(center_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue
                mri_dir = os.path.join(patient_path, 'MRI')
                if not os.path.exists(mri_dir):
                    continue
                for mri_file in os.listdir(mri_dir):
                    if not mri_file.endswith('.png'):
                        continue
                    mri_file_path = os.path.join(mri_dir, mri_file)
                    mri_img = cv2.imread(mri_file_path, cv2.IMREAD_GRAYSCALE)
                    non_zero_coords = cv2.findNonZero(mri_img)
                    if non_zero_coords is not None:
                        x, y, w, h = cv2.boundingRect(non_zero_coords)
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x + w)
                        max_y = max(max_y, y + h)

    min_x = max(min_x - margin, 0)
    min_y = max(min_y - margin, 0)
    max_x += margin
    max_y += margin

    width = max_x - min_x
    height = max_y - min_y

    logging.info(f"Unified bounding box: ({min_x}, {min_y}, {width}, {height})")

    return (min_x, min_y, width, height)

# Function to crop and save slices based on bounding box
def crop_and_save_slice(image_path, output_path, bbox):
    x, y, w, h = bbox
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cropped_image = image[y:y + h, x:x + w]
    cv2.imwrite(output_path, cropped_image)

# Function to process slices for each patient
def process_patient_slices(center_dir, patient_dir, input_base_dir, output_base_dir, bbox):
    center_path = os.path.join(input_base_dir, center_dir)
    patient_path = os.path.join(center_path, patient_dir)
    mri_dir = os.path.join(patient_path, 'MRI')
    mask_dir = os.path.join(patient_path, 'Mask')

    if not os.path.exists(mri_dir) or not os.path.exists(mask_dir):
        return

    output_mri_dir = os.path.join(output_base_dir, center_dir, patient_dir, 'MRI')
    output_mask_dir = os.path.join(output_base_dir, center_dir, patient_dir, 'Mask')

    Path(output_mri_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)

    for mri_file in os.listdir(mri_dir):
        if not mri_file.endswith('.png'):
            continue

        mri_file_path = os.path.join(mri_dir, mri_file)
        mask_file_path = os.path.join(mask_dir, mri_file.replace('_mri_', '_mask_'))

        if not os.path.exists(mask_file_path):
            continue

        output_mri_file_path = os.path.join(output_mri_dir, mri_file)
        output_mask_file_path = os.path.join(output_mask_dir, os.path.basename(mask_file_path))

        crop_and_save_slice(mri_file_path, output_mri_file_path, bbox)
        crop_and_save_slice(mask_file_path, output_mask_file_path, bbox)

# Function to process all patients and crop slices concurrently
def process_all_patients(input_base_dir, output_base_dir, bbox, max_workers=4):
    if not os.path.exists(input_base_dir):
        raise FileNotFoundError(f"Input base directory {input_base_dir} does not exist.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_slices, center_dir, patient_dir, input_base_dir, output_base_dir, bbox)
                   for center_dir in os.listdir(input_base_dir)
                   for patient_dir in os.listdir(os.path.join(input_base_dir, center_dir))
                   if os.path.isdir(os.path.join(input_base_dir, center_dir, patient_dir))]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing future: {e}")

# Function to apply histogram equalization to an image
def histogram_equalization(img):
    return cv2.equalizeHist(img)

# Function to process and save an image after histogram equalization
def process_and_save_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return
        equalized_image = histogram_equalization(image)
        cv2.imwrite(image_path, equalized_image)
        logging.info(f"Processed and saved image: {image_path}")
    except Exception as e:
        logging.error(f"Error processing image: {image_path}, error: {e}")

# Function to process images for a patient
def process_patient_images(center_dir, patient_dir, base_dir):
    patient_path = os.path.join(base_dir, center_dir, patient_dir)
    mri_dir = os.path.join(patient_path, 'MRI')
    mask_dir = os.path.join(patient_path, 'Mask')

    if not os.path.exists(mri_dir) or not os.path.exists(mask_dir):
        return

    for mri_file in os.listdir(mri_dir):
        if mri_file.endswith('.png'):
            process_and_save_image(os.path.join(mri_dir, mri_file))

    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            process_and_save_image(os.path.join(mask_dir, mask_file))

# Function to process all patients in a dataset
def process_all_patients_for_histogram_equalization(base_dir, max_workers=4):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_images, center_dir, patient_dir, base_dir)
                   for center_dir in os.listdir(base_dir)
                   for patient_dir in os.listdir(os.path.join(base_dir, center_dir))
                   if os.path.isdir(os.path.join(base_dir, center_dir, patient_dir))]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing future: {e}")

# Function to check if an image is fully black
def is_fully_black(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.all(image == 0)

# Function to process slices and remove fully black MRI slices
def process_patient_slices_for_black_removal(center_dir, patient_dir, input_base_dir, output_base_dir):
    center_path = os.path.join(input_base_dir, center_dir)
    patient_path = os.path.join(center_path, patient_dir)
    mri_dir = os.path.join(patient_path, 'MRI')
    mask_dir = os.path.join(patient_path, 'Mask')

    if not os.path.exists(mri_dir) or not os.path.exists(mask_dir):
        return

    output_mri_dir = os.path.join(output_base_dir, center_dir, patient_dir, 'MRI')
    output_mask_dir = os.path.join(output_base_dir, center_dir, patient_dir, 'Mask')

    Path(output_mri_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)

    for mri_file in os.listdir(mri_dir):
        if not mri_file.endswith('.png'):
            continue

        mri_file_path = os.path.join(mri_dir, mri_file)
        mask_file_path = os.path.join(mask_dir, mri_file.replace('_mri_', '_mask_'))

        if not os.path.exists(mask_file_path):
            continue

        # Check if the MRI slice is fully black
        if not is_fully_black(mri_file_path):
            # Copy non-black MRI slice and the corresponding mask
            shutil.copy(mri_file_path, os.path.join(output_mri_dir, mri_file))
            shutil.copy(mask_file_path, os.path.join(output_mask_dir, os.path.basename(mask_file_path)))
        else:
            logging.info(f"Removed black MRI slice: {mri_file_path}")

# Function to process slices for black removal concurrently
def process_slices_for_black_removal_concurrently(input_base_dir, output_base_dir, max_workers=4):
    if not os.path.exists(input_base_dir):
        raise FileNotFoundError(f"Input base directory {input_base_dir} does not exist.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_slices_for_black_removal, center_dir, patient_dir, input_base_dir, output_base_dir)
                   for center_dir in os.listdir(input_base_dir)
                   for patient_dir in os.listdir(os.path.join(input_base_dir, center_dir))
                   if os.path.isdir(os.path.join(input_base_dir, center_dir, patient_dir))]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing future: {e}")

# Function to collect patient files
def collect_patient_files(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")
    
    data = []
    for center_dir in os.listdir(base_dir):
        center_path = os.path.join(base_dir, center_dir)
        if not os.path.isdir(center_path):
            continue
        for patient_dir in os.listdir(center_path):
            patient_path = os.path.join(center_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            mri_dir = os.path.join(patient_path, 'MRI')
            mask_dir = os.path.join(patient_path, 'Mask')
            if not os.path.exists(mri_dir) or not os.path.exists(mask_dir):
                continue
            for mri_file in os.listdir(mri_dir):
                if not mri_file.endswith('.png'):
                    continue
                mask_file = mri_file.replace('_mri_', '_mask_')
                mask_file_path = os.path.join(mask_dir, mask_file)
                if not os.path.exists(mask_file_path):
                    continue
                plane = mri_file.split('_')[3]
                data.append({
                    'filename': os.path.join(mri_dir, mri_file),
                    'center': center_dir,
                    'patient_id': patient_dir,
                    'mask': mask_file_path,
                    'plane': plane
                })
    return data

# Function to split patients into training and validation sets
def split_patients(patient_keys, val_size=3):
    random.shuffle(patient_keys)
    val_patients = patient_keys[:val_size]
    train_patients = patient_keys[val_size:]
    return train_patients, val_patients

# Function to log patient split
def log_patient_split(train_patients, val_patients):
    log = []
    for patient in train_patients:
        log.append(f"Training: {patient['center']}, {patient['patient_id']} was chosen.")
    for patient in val_patients:
        log.append(f"Validation: {patient['center']}, {patient['patient_id']} was chosen.")
    with open('/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data/patient_split_log.txt', 'w') as f:
        for line in log:
            f.write(line + '\n')

# Function to write data to CSV
def write_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'center', 'patient_id', 'mask', 'plane']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    setup_preprocess_logging("preprocessing")
    base_dir_train_original = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/original_data/Training'
    base_dir_test_original = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/original_data/Testing'

    output_base_dir_train_preprocessed = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data/Training'
    output_base_dir_test_preprocessed = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data/Testing'

    # Process original training data
    process_all_centers_and_patients(base_dir_train_original, output_base_dir_train_preprocessed, max_workers=16)
    # Process original testing data
    process_all_centers_and_patients(base_dir_test_original, output_base_dir_test_preprocessed, max_workers=16)

    # Find unified bounding box using both training and testing data
    unified_bbox = find_unified_bounding_box([output_base_dir_train_preprocessed, output_base_dir_test_preprocessed])
    logging.info(f"Unified bounding box: {unified_bbox}")

    # Crop slices for training data
    crop_output_base_dir_train = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data/crop/Training'
    crop_output_base_dir_test = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data/crop/Testing'

    # Ensure the directory exists before processing slices for black removal
    if not os.path.exists(crop_output_base_dir_train):
        os.makedirs(crop_output_base_dir_train)
    if not os.path.exists(crop_output_base_dir_test):
        os.makedirs(crop_output_base_dir_test)

    process_all_patients(output_base_dir_train_preprocessed, crop_output_base_dir_train, unified_bbox, max_workers=16)
    process_all_patients(output_base_dir_test_preprocessed, crop_output_base_dir_test, unified_bbox, max_workers=16)

    process_all_patients_for_histogram_equalization(crop_output_base_dir_train, max_workers=16)
    process_all_patients_for_histogram_equalization(crop_output_base_dir_test, max_workers=16)

    # Remove fully black slices for training data
    clean_output_base_dir_train = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data/cleaned/Training'
    clean_output_base_dir_test = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data/cleaned/Testing'

    if not os.path.exists(clean_output_base_dir_train):
        os.makedirs(clean_output_base_dir_train)
    if not os.path.exists(clean_output_base_dir_test):
        os.makedirs(clean_output_base_dir_test)
        
    process_slices_for_black_removal_concurrently(crop_output_base_dir_train, clean_output_base_dir_train, max_workers=16)
    process_slices_for_black_removal_concurrently(crop_output_base_dir_test, clean_output_base_dir_test, max_workers=16)

    # Collect patient files for training/validation
    all_files = collect_patient_files(clean_output_base_dir_train)

    # Group patient files by center and patient_id
    patient_files = {}
    for file_info in all_files:
        key = (file_info['center'], file_info['patient_id'])
        if key not in patient_files:
            patient_files[key] = []
        patient_files[key].append(file_info)

    # Split patients into training and validation sets
    patient_keys = list(patient_files.keys())
    train_patients, val_patients = split_patients(patient_keys, val_size=3)

    # Collect training and validation data
    train_data = [file_info for patient in train_patients for file_info in patient_files[patient]]
    val_data = [file_info for patient in val_patients for file_info in patient_files[patient]]

    # Log patient split
    log_patient_split(
        [{'center': center, 'patient_id': patient_id} for center, patient_id in train_patients],
        [{'center': center, 'patient_id': patient_id} for center, patient_id in val_patients]
    )

    # Write training and validation data to CSV files
    write_csv(train_data, train_csv_path)
    write_csv(val_data, validation_csv_path)

    # Collect all patient files for testing
    test_data = collect_patient_files(clean_output_base_dir_test)

    # Write test data to CSV file
    write_csv(test_data, test_csv_path)
