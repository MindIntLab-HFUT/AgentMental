import pandas as pd
import json
import os
from tqdm import tqdm

def process_phq8_dataset(score_csv_path, transcript_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    scores_df = pd.read_csv(score_csv_path)

    for _, row in tqdm(scores_df.iterrows(), total=len(scores_df), desc="Processing participants"):
        # participant_id = str(row['Participant_ID'])
        participant_id = str(int(row['Participant_ID']))
        transcript_path = os.path.join(transcript_folder, f"{participant_id}_TRANSCRIPT.csv")

        print(f"[处理中] Participant_ID: {participant_id}")
        
        try:
            transcript_df = pd.read_csv(transcript_path, delimiter='\t', dtype={'value': str})
        except FileNotFoundError:
            print(f"Warning: transcript file {transcript_path} not found, skipping this participant")
            continue
        
        real_interview = []
        for _, t_row in transcript_df.iterrows():
            speaker = t_row['speaker']
            content = t_row['value']
            
            if not content or pd.isna(content) or speaker not in ['Ellie', 'Participant']:
                continue
            content = str(content).strip()
            if not content:
                continue

            real_interview.append({
                "roleName": speaker,
                "content": content
            })

        phq8_scores = {
            "PHQ8_Score": int(row['PHQ8_Score']),
            "PHQ8_Binary": int(row['PHQ8_Binary']),
            "items": {
                "PHQ8_NoInterest": int(row['PHQ8_NoInterest']),
                "PHQ8_Depressed": int(row['PHQ8_Depressed']),
                "PHQ8_Sleep": int(row['PHQ8_Sleep']),
                "PHQ8_Tired": int(row['PHQ8_Tired']),
                "PHQ8_Appetite": int(row['PHQ8_Appetite']),
                "PHQ8_Failure": int(row['PHQ8_Failure']),
                "PHQ8_Concentrating": int(row['PHQ8_Concentrating']),
                "PHQ8_Moving": int(row['PHQ8_Moving'])
            }
        }

        result = {
            "Participant_ID": participant_id,
            "real_interview": real_interview,
            "phq8_scores": phq8_scores
        }

        output_path = os.path.join(output_folder, f"{participant_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    SCORE_CSV = "./daic_woz_dataset/dev_transcript_dataset/dev_split_Depression_AVEC2017.csv"
    TRANSCRIPT_DIR = "./daic_woz_dataset/dev_transcript_dataset"
    OUTPUT_DIR = "./processed_dev_daic_woz"

    process_phq8_dataset(SCORE_CSV, TRANSCRIPT_DIR, OUTPUT_DIR)
    print(f"Processing complete! Results saved to {OUTPUT_DIR} folder")