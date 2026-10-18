# train_faces.py - Generate face encodings from cricket player images
import os
import face_recognition
import pickle
import json
import cv2
import numpy as np
from pathlib import Path

# ==== Configuration ====
DATASET_DIR = "data/images/"  # Directory containing player folders
ENCODINGS_PATH = "encodings/encodings.pkl"
PLAYER_DATA_PATH = "player_data.json"
UPSAMPLE_TIMES = 1  # Higher = detect smaller faces (0-3 recommended)
MODEL_TYPE = "hog"  # "hog" for speed, "cnn" for accuracy (needs CUDA for speed)
MAX_WIDTH = 800     # Resize large images for speed
MIN_FACE_SIZE = 50  # Minimum face size in pixels

def normalize_name(name):
    """Normalize player name for consistent matching"""
    return name.lower().replace(" ", "_").replace("-", "_")

def validate_dataset_structure():
    """Validate the dataset directory structure"""
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        print(f"Please create the directory structure:")
        print(f"{DATASET_DIR}")
        print("├── virat_kohli/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("├── ms_dhoni/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        return False
    
    # Check if there are player folders
    player_folders = [f for f in os.listdir(DATASET_DIR) 
                     if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    if not player_folders:
        print(f"❌ No player folders found in {DATASET_DIR}")
        return False
    
    print(f"✅ Found {len(player_folders)} player folders:")
    for folder in player_folders[:5]:  # Show first 5
        image_count = len([f for f in os.listdir(os.path.join(DATASET_DIR, folder)) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   - {folder}: {image_count} images")
    
    if len(player_folders) > 5:
        print(f"   ... and {len(player_folders) - 5} more")
    
    return True

def load_existing_encodings():
    """Load existing encodings if they exist"""
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
                print(f"📂 Loaded existing encodings: {len(data['encodings'])} faces")
                return data
        except Exception as e:
            print(f"⚠️ Could not load existing encodings: {e}")
    
    return {"encodings": [], "names": []}

def get_processed_names(known_encodings):
    """Get set of already processed player names"""
    return set([normalize_name(name.split('_')[0] if '_' in name else name) 
                for name in known_encodings["names"]])

def resize_image_if_needed(image, max_width=MAX_WIDTH):
    """Resize image if it's too large"""
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image

def validate_face_quality(face_location, image_shape):
    """Validate if the detected face is of good quality"""
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top
    
    # Check minimum face size
    if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
        return False, f"Face too small: {face_width}x{face_height}"
    
    # Check if face is too close to edges
    margin = 10
    if left < margin or top < margin or right > image_shape[1] - margin or bottom > image_shape - margin:
        return False, "Face too close to image edges"
    
    return True, "OK"

def process_player_images(player_name, player_folder, known_encodings, skip_existing=True):
    """Process all images for a single player"""
    player_folder_path = os.path.join(DATASET_DIR, player_folder)
    normalized_player = normalize_name(player_name)
    
    if not os.path.isdir(player_folder_path):
        return 0
    
    # Skip if already processed (optional)
    if skip_existing:
        processed_names = get_processed_names(known_encodings)
        if normalized_player in processed_names:
            print(f"⏩ Skipping already processed: {player_name}")
            return 0
    
    print(f"📂 Processing player: {player_name}")
    
    image_files = [f for f in os.listdir(player_folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"   ⚠️ No image files found in {player_folder}")
        return 0
    
    processed_count = 0
    
    for img_name in image_files:
        img_path = os.path.join(player_folder_path, img_name)
        
        try:
            # Load image
            image = face_recognition.load_image_file(img_path)
            
            # Resize if needed
            original_shape = image.shape
            image = resize_image_if_needed(image)
            
            if image.shape != original_shape:
                print(f"   📏 Resized {img_name}: {original_shape[:2]} -> {image.shape[:2]}")
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                image, 
                number_of_times_to_upsample=UPSAMPLE_TIMES, 
                model=MODEL_TYPE
            )
            
            if not face_locations:
                print(f"   🚫 No face found in {img_name}")
                continue
            
            # Process each face found
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            for i, (face_location, encoding) in enumerate(zip(face_locations, face_encodings)):
                # Validate face quality
                is_valid, reason = validate_face_quality(face_location, image.shape)
                
                if not is_valid:
                    print(f"   ❌ Poor quality face in {img_name}: {reason}")
                    continue
                
                # Add to encodings
                face_id = f"{normalized_player}_{processed_count + 1}" if i == 0 else f"{normalized_player}_{processed_count + 1}_{i}"
                
                known_encodings["encodings"].append(encoding)
                known_encodings["names"].append(face_id)
                
                processed_count += 1
                print(f"   ✅ Encoded face {i+1} from {img_name} -> {face_id}")
            
        except Exception as e:
            print(f"   ⚠️ Could not process {img_name}: {e}")
            continue
    
    print(f"   📊 Processed {processed_count} faces for {player_name}")
    return processed_count

def create_player_mapping():
    """Create mapping from normalized names to actual player data"""
    try:
        with open(PLAYER_DATA_PATH, "r", encoding="utf-8") as f:
            player_data = json.load(f)
        
        player_mapping = {}
        for player in player_data:
            normalized = normalize_name(player["name"])
            player_mapping[normalized] = player["name"]
        
        return player_mapping
    except Exception as e:
        print(f"⚠️ Could not load player data: {e}")
        return {}

def generate_training_report(known_encodings, total_processed):
    """Generate and display training report"""
    print("\n" + "="*60)
    print("📊 TRAINING REPORT")
    print("="*60)
    
    # Count faces per player
    player_counts = {}
    for name in known_encodings["names"]:
        base_name = name.split('_')[0] if '_' in name else name
        player_counts[base_name] = player_counts.get(base_name, 0) + 1
    
    print(f"Total players: {len(player_counts)}")
    print(f"Total face encodings: {len(known_encodings['encodings'])}")
    print(f"Newly processed: {total_processed}")
    
    print("\nFaces per player:")
    for player, count in sorted(player_counts.items()):
        print(f"  {player}: {count} faces")
    
    # Quality metrics
    avg_faces_per_player = len(known_encodings['encodings']) / len(player_counts) if player_counts else 0
    print(f"\nAverage faces per player: {avg_faces_per_player:.1f}")
    
    if avg_faces_per_player < 3:
        print("⚠️  Warning: Low number of faces per player. Consider adding more images for better accuracy.")
    elif avg_faces_per_player > 10:
        print("ℹ️  Info: High number of faces per player. This should provide excellent accuracy.")
    else:
        print("✅ Good balance of faces per player.")

def save_encodings(known_encodings):
    """Save encodings to pickle file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
    
    # Save encodings
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(known_encodings, f)
    
    print(f"💾 Saved encodings to {ENCODINGS_PATH}")

def main():
    """Main training function"""
    print("🏏 CricketAI Face Recognition Training")
    print("=====================================")
    
    # Validate dataset structure
    if not validate_dataset_structure():
        return
    
    # Load existing encodings
    known_encodings = load_existing_encodings()
    initial_count = len(known_encodings["encodings"])
    
    # Get player mapping
    player_mapping = create_player_mapping()
    
    # Process each player folder
    total_processed = 0
    
    for player_folder in sorted(os.listdir(DATASET_DIR)):
        player_folder_path = os.path.join(DATASET_DIR, player_folder)
        
        if not os.path.isdir(player_folder_path):
            continue
        
        # Get actual player name from mapping or use folder name
        normalized_folder = normalize_name(player_folder)
        actual_player_name = player_mapping.get(normalized_folder, player_folder.replace('_', ' ').title())
        
        processed = process_player_images(actual_player_name, player_folder, known_encodings)
        total_processed += processed
    
    # Save encodings
    if total_processed > 0 or initial_count == 0:
        save_encodings(known_encodings)
    
    # Generate report
    generate_training_report(known_encodings, total_processed)
    
    print(f"\n✅ Training completed successfully!")
    print(f"Ready to use with CricketAI application.")

if __name__ == "__main__":
    main()
