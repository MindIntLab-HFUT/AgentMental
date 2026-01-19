import json
import logging
import sys

logger = logging.getLogger(__name__)


def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File invalid: {file_path}")
        raise

def load_chatprompt(file_path="scale/chatdemo.json"):
    return load_json_file(file_path)

def load_scoring_standards(file_path="scale/scoring_standards.json"):
    return load_json_file(file_path)

def load_real_data(file_path="real_data.json", scale_name="PHQ-8"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        identifier = data.get("Participant_ID" if scale_name == "PHQ-8" else "video_name", "unknown_identifier")
        real_interview = data.get("real_interview", [])
        scores = data.get("phq8_scores" if scale_name == "PHQ-8" else "scores", {})
        return identifier, real_interview, scores


