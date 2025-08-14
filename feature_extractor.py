import xml.etree.ElementTree as ET
import pandas as pd
import os
import zipfile
import io

# --- Configuration ---
ANNOTATIONS_DIR = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\annotations'
OUTPUT_CSV = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\features.csv'

def parse_single_xml(xml_content):
    """Parses a single annotations.xml content string."""
    root = ET.fromstring(xml_content)
    
    task_name = root.find('.//task/name').text
    video_type = 'shoplifting' if 'shoplifting' in task_name.lower() else 'normal'
    
    frames_data = []
    for image_elem in root.findall('image'):
        frame_info = {
            'task_name': task_name,
            'video_type': video_type,
            'frame_id': int(image_elem.get('id')),
            'image_name': image_elem.get('name'),
            'person_boxes': [],
            'product_boxes': [],
            'bag_boxes': []
        }

        for box in image_elem.findall('box'):
            box_data = {
                'label': box.get('label'),
                'xtl': float(box.get('xtl')),
                'ytl': float(box.get('ytl')),
                'xbr': float(box.get('xbr')),
                'ybr': float(box.get('ybr')),
                'occluded': box.get('occluded') == '1'
            }
            
            if box_data['label'] == 'person':
                frame_info['person_boxes'].append(box_data)
            elif box_data['label'] == 'object':
                frame_info['product_boxes'].append(box_data)
            elif box_data['label'] == 'bag':
                frame_info['bag_boxes'].append(box_data)
        
        frames_data.append(frame_info)
        
    return frames_data

def calculate_features(df):
    """Calculates features from the parsed data."""
    print("Calculating features...")
    features = []

    for _, row in df.iterrows():
        num_people = len(row['person_boxes'])
        num_products = len(row['product_boxes'])
        num_bags = len(row['bag_boxes'])

        min_person_product_dist = float('inf')
        product_in_bag_occluded = 0

        if num_people > 0 and num_products > 0:
            for p_box in row['person_boxes']:
                p_center = ((p_box['xtl'] + p_box['xbr']) / 2, (p_box['ytl'] + p_box['ybr']) / 2)
                for pr_box in row['product_boxes']:
                    pr_center = ((pr_box['xtl'] + pr_box['xbr']) / 2, (pr_box['ytl'] + pr_box['ybr']) / 2)
                    dist = ((p_center[0] - pr_center[0])**2 + (p_center[1] - pr_center[1])**2)**0.5
                    if dist < min_person_product_dist:
                        min_person_product_dist = dist

        if num_bags > 0 and num_products > 0:
            for b_box in row['bag_boxes']:
                for pr_box in row['product_boxes']:
                    pr_center_x = (pr_box['xtl'] + pr_box['xbr']) / 2
                    pr_center_y = (pr_box['ytl'] + pr_box['ybr']) / 2
                    if (b_box['xtl'] < pr_center_x < b_box['xbr'] and \
                        b_box['ytl'] < pr_center_y < b_box['ybr'] and \
                        pr_box['occluded']):
                        product_in_bag_occluded = 1
                        break
                if product_in_bag_occluded: break
        
        is_shoplifting_event = 1 if row['video_type'] == 'shoplifting' else 0

        features.append({
            'task_name': row['task_name'],
            'frame_id': row['frame_id'],
            'num_people': num_people,
            'num_products': num_products,
            'num_bags': num_bags,
            'min_person_product_dist': min_person_product_dist if min_person_product_dist != float('inf') else -1,
            'product_in_bag_occluded': product_in_bag_occluded,
            'is_shoplifting_event': is_shoplifting_event
        })

    return pd.DataFrame(features)

if __name__ == '__main__':
    all_files_data = []
    zip_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.zip')]
    
    if not zip_files:
        print(f"Error: No .zip files found in {ANNOTATIONS_DIR}")
    else:
        print(f"Found {len(zip_files)} zip files to process.")
        for zip_filename in zip_files:
            zip_filepath = os.path.join(ANNOTATIONS_DIR, zip_filename)
            print(f"Processing {zip_filename}...")
            try:
                with zipfile.ZipFile(zip_filepath, 'r') as zf:
                    with zf.open('annotations.xml', 'r') as xml_file:
                        xml_content = xml_file.read()
                        all_files_data.extend(parse_single_xml(xml_content))
            except Exception as e:
                print(f"Could not process {zip_filename}. Error: {e}")

        if all_files_data:
            master_df = pd.DataFrame(all_files_data)
            feature_df = calculate_features(master_df)
            
            feature_df.to_csv(OUTPUT_CSV, index=False)
            
            print(f"\nFeature extraction complete.")
            print(f"CSV file saved to: {OUTPUT_CSV}")
            print(f"Total frames processed: {len(feature_df)}")
            print(f"Total videos processed: {len(zip_files)}")
            print("\nSample of the final data:")
            print(feature_df.head())
            print("\nValue counts for 'is_shoplifting_event':")
            print(feature_df['is_shoplifting_event'].value_counts())
        else:
            print("No data was successfully parsed from any of the zip files.")
