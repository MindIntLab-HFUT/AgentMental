import os
import sys
import logging
import pandas as pd
from config import get_llm_config
from data_load import load_chatprompt, load_scoring_standards, load_real_data
from logging_setup import setup_logging, initialize_dialog_log, dialog_print, close_dialog_log
from utils import get_valid_input, choose_mode
from assessment import process_single_file


if __name__ == "__main__":
    logger = setup_logging()
    initialize_dialog_log()
    logger.info("Psychological assessment program started.")

    # data_dir = "data/depression_data_111"
    data_dir = "../data/processed_train_daic_woz"
    if not os.path.exists(data_dir):
        logger.error(f"Data folder {data_dir} does not exist.")
        dialog_print(f"Error: Data folder {data_dir} does not exist.")
        sys.exit(1)

    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
    if not json_files:
        logger.error(f"No JSON files found in folder {data_dir}.")
        dialog_print(f"Error: No JSON files found in folder {data_dir}.")
        sys.exit(1)

    logger.info(f"Found {len(json_files)} JSON files in folder {data_dir}.")
    dialog_print(f"Found {len(json_files)} JSON files in folder {data_dir}.")

    scoring_standards = load_scoring_standards("../scales/scoring_standards.json")

    available_scales = list(scoring_standards.keys())
    dialog_print("Please choose the psychological assessment scale to use:")
    for idx, scale in enumerate(available_scales, 1):
        print(f"{idx}. {scale}")

    while True:
        scale_choice = get_valid_input(f"Enter the scale number (1-{len(available_scales)}): ")
        if scale_choice.isdigit():
            scale_idx = int(scale_choice)
            if 1 <= scale_idx <= len(available_scales):
                selected_scale = available_scales[scale_idx - 1]
                break
        dialog_print("Invalid choice, please try again.")

    logger.info(f"Selected scale: {selected_scale}")
    dialog_print(f"You selected scale: {selected_scale}")

    if selected_scale == "HAMA":
        chatprompt = load_chatprompt("HAMA.json")
    elif selected_scale == "HAMD-17":
        chatprompt = load_chatprompt("HAMD-17.json")
    elif selected_scale == "PHQ-8":
        chatprompt = load_chatprompt("../scales/PHQ-8.json")
    else:
        dialog_print("Unsupported scale type.")
        sys.exit(1)

    mode_choice = choose_mode()
    if mode_choice == "2":
        print("Automated test mode enabled.")
        automated = True
    else:
        automated = False

    csv_file_path = "evaluation/72b.csv"
    for json_file in json_files:
        process_single_file(json_file, scoring_standards, chatprompt, selected_scale, mode_choice, csv_file_path, automated)

    close_dialog_log()
