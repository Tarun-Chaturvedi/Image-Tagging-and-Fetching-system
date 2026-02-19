import os
import cv2
import face_recognition
import numpy as np
from utils import calculate_sha256
from database import init_db, image_exists, insert_image, insert_tag, get_or_create_profile, link_face_to_image
from detector import detect_objects

def scan_folder(folder_path):
    print(f"Starting advanced scan of: {folder_path}")
    init_db() 
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.lower().endswith(valid_extensions):
                full_path = os.path.join(root, name)
                
                img_hash = calculate_sha256(full_path)
                existing_id = image_exists(img_hash)
                
                if existing_id:
                    continue
                
                print(f"[+] Analyzing: {name}")
                
                # 1. Run YOLO
                tags = detect_objects(full_path) 
                
                # 2. Save Image & General Tags
                img_id = insert_image(full_path, img_hash)
                for item in tags:
                    insert_tag(img_id, item['label'], item['confidence'])

                # 3. Handle Person Profiles
                if any(t['label'] == 'person' for t in tags):
                    process_faces(full_path, img_id)

    print("Scan complete!")

def process_faces(image_path, img_id):
    """Crops faces and links them to profiles in the DB."""
    image = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img, face_locations)
    
    for encoding, location in zip(encodings, face_locations):
        # Match against database or create new profile
        profile_id = get_or_create_profile(encoding)
        

        link_face_to_image(img_id, profile_id, encoding, location)

if __name__ == "__main__":
    scan_folder("./my_images")
