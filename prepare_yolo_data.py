
import os
import zipfile
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
ANNOTATIONS_DIR = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\annotations'
FRAMES_DIR = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\dataset\frames'
OUTPUT_DIR = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\yolo_dataset'

# Define your custom classes
CLASS_NAMES = ['person', 'object', 'bag']

def convert_cvat_to_yolo(xml_path, class_names):
    """Converts a single CVAT XML to a list of YOLO formatted strings."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_annotations = {}

    image_width = int(root.find('image').get('width'))
    image_height = int(root.find('image').get('height'))

    for image_tag in root.findall('image'):
        image_name = image_tag.get('name')
        frame_annotations = []
        for box_tag in image_tag.findall('box'):
            label = box_tag.get('label')
            if label not in class_names:
                continue
            
            class_id = class_names.index(label)
            xtl = float(box_tag.get('xtl'))
            ytl = float(box_tag.get('ytl'))
            xbr = float(box_tag.get('xbr'))
            ybr = float(box_tag.get('ybr'))

            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (xtl + xbr) / 2
            y_center = (ytl + ybr) / 2
            width = xbr - xtl
            height = ybr - ytl

            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = width / image_width
            height_norm = height / image_height

            frame_annotations.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
        
        if frame_annotations:
            yolo_annotations[image_name] = "\n".join(frame_annotations)
            
    return yolo_annotations

def main():
    print("Starting YOLO data preparation...")

    # --- 1. Setup Directories ---
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Create train/val splits for images and labels
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'))
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'))
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'))
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'))
    print(f"Created directory structure at {OUTPUT_DIR}")

    # --- 2. Process Annotations ---
    zip_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.zip')]
    all_image_paths = []
    all_annotations = {}

    for zip_filename in zip_files:
        zip_filepath = os.path.join(ANNOTATIONS_DIR, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            with zf.open('annotations.xml', 'r') as xml_file:
                # The XML file needs to be saved temporarily to be parsed by path
                temp_xml_path = os.path.join(ANNOTATIONS_DIR, 'temp_annotations.xml')
                with open(temp_xml_path, 'wb') as f:
                    f.write(xml_file.read())
                
                video_annotations = convert_cvat_to_yolo(temp_xml_path, CLASS_NAMES)
                all_annotations.update(video_annotations)
                os.remove(temp_xml_path)

    print(f"Processed {len(zip_files)} annotation files.")

    # --- 3. Create Train/Val Split ---
    # Find all source image paths from the frames directory
    for video_type in ['normal', 'shoplifting']:
        video_type_path = os.path.join(FRAMES_DIR, video_type)
        if os.path.isdir(video_type_path):
            for item_name in os.listdir(video_type_path):
                item_path = os.path.join(video_type_path, item_name)
                if os.path.isdir(item_path):
                    # This handles the case where frames are in subdirectories (e.g., for shoplifting videos)
                    for frame_filename in os.listdir(item_path):
                        if frame_filename.endswith('.jpg'):
                            all_image_paths.append(os.path.join(item_path, frame_filename))
                elif item_name.endswith('.jpg'):
                    # This handles the case where frames are directly in the folder (e.g., for normal videos)
                    all_image_paths.append(item_path)

    # Filter only the images that have annotations
    annotated_image_paths = [p for p in all_image_paths if os.path.basename(p) in all_annotations]
    
    train_paths, val_paths = train_test_split(annotated_image_paths, test_size=0.2, random_state=42)
    print(f"Splitting data: {len(train_paths)} training images, {len(val_paths)} validation images.")

    # --- 4. Copy Files and Write Labels ---
    for split, paths in [('train', train_paths), ('val', val_paths)]:
        for img_path in paths:
            # Copy image
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, 'images', split))
            
            # Write label file
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_content = all_annotations.get(img_filename)
            
            if label_content:
                with open(os.path.join(OUTPUT_DIR, 'labels', split, label_filename), 'w') as f:
                    f.write(label_content)

    print("\nData preparation complete!")
    print(f"Your YOLO dataset is ready at: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
