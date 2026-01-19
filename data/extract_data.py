import os
import csv
import re
import zipfile
import shutil

# Configuration parameters
TEST_CSV = ""  # dataset CSV filename
SOURCE_DIR = ""  # Directory containing ZIP archives
TARGET_DIR = ""  # Extraction destination
TRANSCRIPT_DIR = ""  # Directory for transcript files

def normalize_zip_name(participant_id):
    spaced_name = f"{participant_id} P.zip"
    underscored_name = f"{participant_id}_P.zip"
    dotted_name = f"{participant_id}. P.zip"
    lowercase_name = f"{participant_id}_p.zip"
    
    return [spaced_name, underscored_name, dotted_name, lowercase_name]

def find_matching_zip(participant_id):
    possible_names = normalize_zip_name(participant_id)
    
    for filename in possible_names:
        filepath = os.path.join(SOURCE_DIR, filename)
        if os.path.exists(filepath):
            return filepath

    for filename in os.listdir(SOURCE_DIR):
        if filename.lower().endswith('.zip'):
            match = re.search(r'(\d{3})', filename)
            if match:
                file_id = int(match.group(1))
                if file_id == participant_id:
                    return os.path.join(SOURCE_DIR, filename)
    return None

def extract_zip(zip_path, target_dir):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"✓ Extraction succeeded: {os.path.basename(zip_path)}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {os.path.basename(zip_path)} - {str(e)}")
        return False

def copy_transcript_files(participant_id, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    patterns = [
        f"{participant_id}_TRANSCRIPT.csv",
        f"{participant_id} TRANSCRIPT.csv",
        f"{participant_id}_TRANSCRIPT.CSV",
        f"{participant_id}_transcript.csv"
    ]
    
    found = False
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if any(file.lower() == pattern.lower() for pattern in patterns):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(target_dir, file)

                shutil.copy2(src_path, dest_path)
                print(f"✓ Copied transcript: {file} → {os.path.basename(target_dir)}")
                found = True
                break
    if not found:
        print(f"✗ Transcript not found for participant {participant_id}")
    return found

def process_test_set():
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    
    participant_ids = []
    try:
        with open(TEST_CSV, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if row:  
                    participant_ids.append(int(row[0]))
        print(f"Loaded {len(participant_ids)} participant IDs from CSV")
    except Exception as e:
        print(f"Failed to read CSV file: {str(e)}")
        return

    success_count = 0
    transcript_count = 0
    missing_zip_count = 0
    missing_transcript_count = 0
    
    for pid in participant_ids:
        print(f"\nProcessing participant ID: {pid}")
        zip_path = find_matching_zip(pid)
        
        if zip_path:
            if extract_zip(zip_path, TARGET_DIR):
                success_count += 1

                if copy_transcript_files(pid, TARGET_DIR, TRANSCRIPT_DIR):
                    transcript_count += 1
                else:
                    missing_transcript_count += 1
            else:
                missing_zip_count += 1
        else:
            print(f"✗ ZIP archive not found for participant {pid}")
            missing_zip_count += 1

    print("\n" + "=" * 50)
    print(f"Processing complete! Summary:")
    print(f"Total participants: {len(participant_ids)}")
    print(f"Successfully extracted: {success_count}")
    print(f"Transcripts found: {transcript_count}")
    print(f"Missing ZIP archives: {missing_zip_count}")
    print(f"Missing transcripts: {missing_transcript_count}")
    print(f"Extraction directory: {os.path.abspath(TARGET_DIR)}")
    print(f"Transcript directory: {os.path.abspath(TRANSCRIPT_DIR)}")

    try:
        shutil.copy2(TEST_CSV, TRANSCRIPT_DIR)
        print(f"Copied CSV file to transcript directory")
    except Exception as e:
        print(f"Failed to copy CSV file: {str(e)}")

if __name__ == "__main__":
    process_test_set()