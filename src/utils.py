import re
import json
import os
import sys
import contextlib
import logging
import pandas as pd
from data_load import load_scoring_standards
from logging_setup import dialog_print
from config import get_llm_config
from autogen.token_count_utils import count_token

logger = logging.getLogger(__name__)
llm_config = get_llm_config()


def parse_personal_info(response):
    gender_options = {"male", "female", "other", "prefer not to say"}
    normalized = re.sub(r'[；;，,、\t\n\r\s]+', ',', response)
    parts = [part.strip() for part in normalized.split(',') if part.strip()]

    age = None
    gender = None
    occupation = None
    occupation_match = re.search(r'occupation[:：是\s]*(.+)', response, re.IGNORECASE)

    if occupation_match:
        occupation = occupation_match.group(1).strip()

    if occupation is None and 'occupation' in response:
        try:
            parts = response.split('occupation')
            if len(parts) > 1:
                occupation_part = parts[1]
                if occupation_part.startswith('：'):
                    occupation_part = occupation_part[1:]
                elif occupation_part.startswith(':'):
                    occupation_part = occupation_part[1:]
                occupation = re.split(r'[,\s，。]', occupation_part.strip())[0]
        except Exception:
            occupation = None

    for part in parts:
        age_match = re.match(r'age[:：=]?\s*(\d+)', part, re.IGNORECASE)
        gender_match = re.match(r'gender[:：=]?\s*(' + '|'.join(gender_options) + ')', part, re.IGNORECASE)

        if age_match:
            age = int(age_match.group(1))
            continue
        if gender_match:
            gender = gender_match.group(1)
            continue
        if age is None and part.isdigit():
            potential_age = int(part)
            if 0 < potential_age <= 120:
                age = potential_age
                continue
        if gender is None and part in gender_options:
            gender = part
            continue
    if age and gender and occupation:
        return age, gender, occupation
    else:
        return None, None, None


def get_valid_input(prompt_text):
    while True:
        try:
            user_input = input(prompt_text).strip()
            if user_input:
                return user_input
            else:
                print("Empty input, please re-enter.")
        except EOFError:
            print("\nInput complete.")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nProgram interrupted.")
            sys.exit(0)

def choose_mode():
    print("\nPlease select a run mode:")
    print("1. Manual interactive mode")
    print("2. Automated test mode")
    while True:
        mode_choice = get_valid_input("Enter the mode number (1 or 2): ")
        if mode_choice in {"1", "2"}:
            return mode_choice
        print("Invalid choice, please try again.")


def categorize_score(total_score, scale_name):
    if scale_name == "PHQ-8":
        if total_score >= 10:
            return 1  
        else:
            return 0 
    else:
        return "Unknown symptom level"


def extract_score(text):
    numbers = re.findall(r"\d+", text)
    if len(numbers) > 0:
        return int(numbers[0])
    else:
        return 0


def extract_score_and_summary(text, scale_name):
    try:
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[: -len("```")].strip()

        data = json.loads(text)
        score = data.get("score", 0)
        summary = data.get("summary", "")

        max_score = 3 if scale_name == "PHQ-8" else 4
        if isinstance(score, int) and 0 <= score <= max_score:
            return score, summary
        else:
            print("Score is out of valid range; defaulting to 0.")
            return 0, summary
    except json.JSONDecodeError:
        print("Unable to parse the scoring agent's output. Please ensure it follows JSON format.")
        return 0, ""


def extract_summary_and_updated_scores(text, scale_name):
    try:
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[: -len("```")].strip()

        data = json.loads(text)
        summary = data.get("summary", "")
        updated_scores = data.get("updated_scores", {})
        
        max_score = 3 if scale_name == "PHQ-8" else 4
        valid_updated_scores = {}

        for topic, score_data in updated_scores.items():
            if isinstance(score_data, dict):
                score = score_data.get("score")
                reason = score_data.get("reason", "")
                if isinstance(score, int) and 0 <= score <= max_score:
                    valid_updated_scores[topic] = {"score": score, "reason": reason}
                else:
                    print(f"Invalid updated score: topic='{topic}', score='{score}'")
            else:
                print(f"Invalid format in updated_scores: topic='{topic}', data='{score_data}'")

        return summary, valid_updated_scores
    except json.JSONDecodeError:
        print("Unable to parse SummaryAgent's output. Please ensure it follows JSON format.")
        return "", {}



def generate_score_table(memory_graph, topics, symptom_level):
    table_data = []
    for topic in topics:
        if memory_graph.graph.has_node(topic):
            node_attrs = memory_graph.graph.nodes[topic]
            if node_attrs.get('status') == 'completed':
                initial_score = node_attrs.get('score', 0)
                updated_score = node_attrs.get('updated_score')
                final_score = updated_score if updated_score is not None else initial_score
                
                table_data.append({
                    "topic": topic,
                    "initial_score": initial_score,
                    "final_score": final_score
                })

    total_score = sum(item["final_score"] for item in table_data)

    df = pd.DataFrame({
        "No.": range(1, len(table_data) + 1),
        "Item": [item["topic"] for item in table_data],
        "Initial Score": [item["initial_score"] for item in table_data],
        "Final Score": [item["final_score"] for item in table_data]
    })

    df.loc[len(df.index)] = ["", "Total Score", "", total_score]
    df.loc[len(df.index)] = ["", "Symptom Level", "", symptom_level]

    return df.to_markdown(index=False)

def generate_report(report_table, summary, scale_name):
    report = f"# Psychological Scale Assessment Report\n\n"
    report += f"## Assessment Scale: {scale_name}\n\n"
    report += "## Score Table\n\n"
    report += f"{report_table}\n\n"
    report += "## Summary and Recommendations\n\n"
    report += f"{summary}\n"
    return report


def custom_speaker_selection_func(last_speaker, groupchat):
    messages = groupchat.messages
    last_message = messages[-1]["content"]
    match = re.search(r"Next speaker: (\w+)", last_message)
    if match:
        target_name = match.group(1)
        for agent in groupchat.agents:
            if agent.name == target_name:
                return agent


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def makerequest(group_chat_manager, user_proxy, agent, prompt):
    full_prompt = f"Next speaker: {agent.name}\n{prompt}"
    logger.info(f"Prompt sent to {agent.name}: {full_prompt}")

    MODEL_MAX_CONTEXT = 32768
    MAX_COMPLETION_TOKENS = llm_config.get("max_tokens", 4096)
    SAFETY_BUFFER = 8192 
    TOKEN_THRESHOLD = MODEL_MAX_CONTEXT - MAX_COMPLETION_TOKENS - SAFETY_BUFFER - 4096

    messages_to_send = group_chat_manager.groupchat.messages
    current_messages_tokens = count_token(messages_to_send)
    current_prompt_tokens = count_token(full_prompt)
    current_tokens = current_messages_tokens + current_prompt_tokens
    
    while current_tokens > TOKEN_THRESHOLD:
        if len(messages_to_send) > 1:
            removed_message = messages_to_send.pop(0)
            current_tokens = count_token(messages_to_send) + current_prompt_tokens
            logger.warning(f"Message history too long; removed one old message. Current tokens: {current_tokens}, threshold: {TOKEN_THRESHOLD}")
        else:
            logger.error("Message history contains only one message but still exceeds token threshold, unable to trim further.")
            break
    for ag in group_chat_manager.groupchat.agents:
        ag.chat_messages[group_chat_manager] = group_chat_manager.groupchat.messages

    try:
        with suppress_output():
            response = user_proxy.initiate_chat(
                group_chat_manager,
                message=full_prompt,
                max_turns=1,
                clear_history=False
            )
        # print(response.chat_history)
        original_response_text = response.chat_history[-1]["content"].strip()
        logger.info(f"Raw response from {agent.name}: {original_response_text}")
        
        think_index = original_response_text.find("</think>")
        if think_index != -1:
            processed_response_text = original_response_text[think_index + len("</think>"):].strip()
        else:
            processed_response_text = original_response_text
        logger.info(f"Processed response from {agent.name}: {processed_response_text}")
        group_chat_manager.groupchat.messages[-1]["content"] = processed_response_text
        return processed_response_text
    
    except Exception as e:
        logger.exception(f"Error while calling {agent.name}: %s", e)
        return None
    

def is_necessary(necessity_score, asked_questions):
    if necessity_score == 2:
        logger.info("Necessity score == 2, continue in-depth questioning.")
        dialog_print("Necessity score == 2, continue in-depth questioning.")
        return True
    elif necessity_score == 1:
        if asked_questions < 2:
            logger.info("Necessity score == 1, and the number of questions asked has not reached 2, continue questioning.")
            dialog_print("Necessity score == 1, and the number of questions asked has not reached 2, continue questioning.")
            return True
        else:
            logger.info("Necessity score == 1, and the number of questions asked has reached 2, stop questioning.")
            dialog_print("Necessity score == 1, and the number of questions asked has reached 2, stop questioning.")
            return False
    else:
        logger.info("Necessity score == 0, stop questioning.")
        dialog_print("Necessity score == 0, stop questioning.")
        return False
    

def save_assessment_results(identifier, overall_score, symptom_level, updated_scores, csv_file="depression.csv", scale_name="PHQ-8"):
    data = {
        "identifier": identifier,
        "classes": symptom_level,
        "total": overall_score
    }
    score_values = {topic: details["score"] for topic, details in updated_scores.items()}
    max_items = 8
    for idx, (topic, score) in enumerate(score_values.items(), 1):
        item_key = f"item{idx}"
        data[item_key] = score
    for i in range(1, max_items + 1):
        item_key = f"item{i}"
        if item_key not in data:
            data[item_key] = 0 

    if os.path.isfile(csv_file):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            if identifier in df['identifier'].values:
                index = df.index[df['identifier'] == identifier].tolist()[0]
                for key, value in data.items():
                    df.at[index, key] = value
                logger.info(f"Updated evaluation results for {identifier}.")
            else:
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                logger.info(f"Appended evaluation results for {identifier}.")
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file}: {e}")
            print("Error reading CSV file; please check the logs.")
            return
    else:
        columns = ["identifier", "total", "classes"] + [f"item{i}" for i in range(1, max_items + 1)]
        df = pd.DataFrame([data], columns=columns)
        logger.info(f"A new CSV file {csv_file} has been created and the evaluation results have been saved.")
    try:
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Evaluation results saved to {csv_file}")
    except Exception as e:
        logger.error(f"Error saving CSV file {csv_file}: {e}")
        print("Error saving CSV file; please check the logs.")

def is_file_already_evaluated(identifier, csv_file_path):
    if not os.path.isfile(csv_file_path):
        return False  
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        normalized_identifiers = df['identifier'].astype(str).str.strip()
        return str(identifier).strip() in normalized_identifiers.values
    except Exception as e:
        logger.error(f"Error checking CSV file {csv_file_path}: {e}")
        return False  

