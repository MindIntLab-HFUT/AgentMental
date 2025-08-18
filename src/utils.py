import re
import json
import os
import sys
import contextlib
import logging
import pandas as pd
from data_load import load_scoring_standards
from logging_setup import dialog_print

logger = logging.getLogger(__name__)


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
                print("输入不能为空，请重新输入。")
        except EOFError:
            print("\n输入结束。")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\n程序中断。")
            sys.exit(0)

def choose_mode():
    print("\n请选择运行模式：")
    print("1. 人工交互模式")
    print("2. 自动化测试模式")
    while True:
        mode_choice = get_valid_input("请输入选择的模式编号（1或2）： ")
        if mode_choice in {"1", "2"}:
            return mode_choice
        print("无效的选择，请重新输入。")


def categorize_score(total_score, scale_name):
    if scale_name == "HAMA":
        if total_score >= 29:
            return "严重有焦虑"
        elif total_score >= 21:
            return "明显有焦虑"
        elif total_score >= 14:
            return "肯定有焦虑"
        elif total_score >= 7:
            return "可能有焦虑症"
        else:
            return "无焦虑"
    elif scale_name == "HAMD-17":
        if total_score > 24:
            return "严重抑郁症"
        elif total_score >= 17:
            return "肯定有抑郁症"
        elif total_score >= 7:
            return "可能有抑郁症"
        else:
            return "正常"
    elif scale_name == "PHQ-8":
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
            print("评分不在有效范围内，默认评分为0。")
            return 0, summary
    except json.JSONDecodeError:
        print("无法解析评分智能体的输出，请确保其遵循JSON格式。")
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
                    print(f"无效的更新分数：主题='{topic}', 分数='{score}'")
            else:
                print(f"无效的 updated_scores 格式：主题='{topic}', 数据='{score_data}'")

        return summary, valid_updated_scores
    except json.JSONDecodeError:
        print("无法解析 SummaryAgent 的输出，请确保其遵循JSON格式。")
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


def makerequest(agent, prompt):

    try:
        logger.info(f"向{agent.name}发送消息：{prompt}")
        with suppress_output():
            response = agent.initiate_chat(
                recipient=agent,
                message=prompt,
                max_turns=1
            )
        response_text = response.chat_history[-1]["content"].strip()
        logger.info(f"{agent.name}的响应：{response_text}")
        
        think_index = response_text.find("</think>")
        if think_index != -1:
            response_text = response_text[think_index + len("</think>") :].strip()

        return response_text
    
    except Exception as e:
        logger.exception(f"调用{agent.name}时发生错误：%s", e)
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
    

def generate_mock_response(question, topic, profile, identification, real_interview, scale_scores, scoring_standard=None, current_topic_history=None, depth=0, clear_memory=False, scale_name="PHQ-8"):
    profile_str = json.dumps(profile, indent=2, ensure_ascii=False)
    interview_history = ""
    for para in real_interview:
        role = para.get("roleName", "Unknown role")
        content = para.get("content", "")
        interview_history += f"{role}: {content}\n"

    if clear_memory:
        prompt = "Please forget all previous conversation content and start a new session. Limit your response to 10 words."
        logger.info("Clearing API's conversation memory...")

    elif identification == "":
        prompt = f"""Please answer the following question:\n{question}\n
Please provide your basic information (age, gender, occupation) in the format 'age:<age>, gender:<gender>, occupation:<occupation>' (e.g., age:25, gender:male, occupation:engineer).
Ensure the response is concise and follows the requested format without adding anything else.\n
"""
        logger.info(f"生成模拟回复的提示：{prompt}")
    else:
        if depth == 0:
            prompt = f"""Please answer the following question:\n{question}\nPlease provide a truthful and reasonable answer based on your real interview dialogue and your profile, with no more than 50 words:"""
        else:
            topic_history_str = ""
            if current_topic_history:
                for qa in current_topic_history:
                    topic_history_str += f"BingTang's Question: {qa['question']}\nUser's Response: {qa['response']}\n"
            prompt = f"""Please answer the following in-depth question:\n{question}\n
Please provide a truthful and reasonable answer based on your profile, the interview dialogue and all your previous responses. Do not fabricate symptom situations in your response.
When the question content exceeds the scope of the interview dialogue and cannot be accurately answered, you may choose to answer 'not sure' or a similar expression, but use it sparingly and provide a reason that fits the patient's information.
Limit your response to 50 words.
"""
        logger.info(f"生成模拟回复的提示：{prompt}")
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="",
            api_key=""
        )

        system_prompt = f"""You are speaking with a psychological assistant named 'BingTang', and you are the client in this conversation with the following personal information and previous interview dialogue:
{profile_str}\n
{interview_history}

You should follow the provided information to act as a client in the conversation. Your responses should be coherent and avoid repeating previous utterances.
Your response should ONLY include what the Client should say, in a natural, first-person tone.
""" 
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        response = completion.choices[0].message.content
        logger.info(f"生成的模拟回复：{response}")

        think_index = response.find("</think>")
        if think_index != -1:
            response = response[think_index + len("</think>") :].strip()
        return response
    except Exception as e:
        logger.exception(f"调用 OpenAI API 生成回复时发生错误：{e}")
        return "抱歉，我现在无法回答这个问题。"



def save_assessment_results(identifier, overall_score, symptom_level, updated_scores, csv_file="depression.csv", scale_name="HAMD-17"):
    data = {
        "identifier": identifier,
        "classes": symptom_level,
        "total": overall_score
    }

    score_values = {topic: details["score"] for topic, details in updated_scores.items()}
    max_items = 17 if scale_name == "HAMD-17" else 8 if scale_name == "PHQ-8" else 14
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
                logger.info(f"已更新 {identifier} 的评估结果。")
            else:
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                logger.info(f"已追加 {identifier} 的评估结果。")
        except Exception as e:
            logger.error(f"读取CSV文件 {csv_file} 时发生错误：{e}")
            print(f"读取CSV文件时发生错误，请检查日志。")
            return
    else:
        columns = ["identifier", "total", "classes"] + [f"item{i}" for i in range(1, max_items + 1)]
        df = pd.DataFrame([data], columns=columns)
        logger.info(f"已创建新的CSV文件 {csv_file} 并保存评估结果。")
    try:
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"评估结果已保存到 {csv_file}")
    except Exception as e:
        logger.error(f"保存CSV文件 {csv_file} 时发生错误：{e}")
        print(f"保存CSV文件时发生错误，请检查日志。")

def is_file_already_evaluated(identifier, csv_file_path):
    if not os.path.isfile(csv_file_path):
        return False  

    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        normalized_identifiers = df['identifier'].astype(str).str.strip()
        return str(identifier).strip() in normalized_identifiers.values
    except Exception as e:
        logger.error(f"检查CSV文件 {csv_file_path} 时发生错误：{e}")
        return False  

