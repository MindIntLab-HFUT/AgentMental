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
    
    gender_options = {"男", "女", "其他", "不愿透露"}

    normalized = re.sub(r'[；;，,、\t\n\r\s]+', ',', response)

    parts = [part.strip() for part in normalized.split(',') if part.strip()]

    age = None
    gender = None
    occupation = None

    for part in parts:
        age_match = re.match(r'年龄[:：=]?\s*(\d+)', part)
        gender_match = re.match(r'性别[:：=]?\s*(' + '|'.join(gender_options) + ')', part)
        occupation_match = re.match(r'职业[:：=]?\s*(.+)', part)

        if age_match:
            age = int(age_match.group(1))
            continue

        if gender_match:
            gender = gender_match.group(1)
            continue

        if occupation_match:
            occupation = occupation_match.group(1).strip()
            continue

        if age is None and part.isdigit():
            potential_age = int(part)
            if 0 < potential_age <= 120:
                age = potential_age
                continue

        if gender is None and part in gender_options:
            gender = part
            continue

        if occupation is None:
            occupation = part
            continue

    if age and gender and occupation:
        return age, gender, occupation
    else:
        return None, None, None


# 获取有效的用户输入
def get_valid_input(prompt_text):
    while True:
        try:
            user_input = input(prompt_text).strip()
            if user_input:
                return user_input
            else:
                print("输入不能为空，请重新输入。")

        except UnicodeDecodeError:
            print("输入包含无效字符，请使用标准字符输入。")
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


# 症状判断函数
def categorize_score(total_score, scale_name):
    """
    根据总评分和量表类型判断症状的严重程度。
    """
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
    else:
        return "未知症状级别"


def extract_score(text):
    numbers = re.findall(r"\d+", text)
    if len(numbers) > 0:
        return int(numbers[0])
    else:
        return 0


def extract_score_and_summary(text):
    """
    从评分结果文本中提取评分和总结。
    假设评分智能体输出严格的JSON格式。
    """
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
        if isinstance(score, int) and 0 <= score <= 4:
            return score, summary
        else:
            print("评分不在有效范围内，默认评分为0。")
            return 0, summary
    except json.JSONDecodeError:
        print("无法解析评分智能体的输出，请确保其遵循JSON格式。")
        return 0, ""


# 提取总结以及更新分数
def extract_summary_and_updated_scores(text):
    """
    从 SummaryAgent 的输出中提取总结和更新后的分数。
    假设 SummaryAgent 严格遵循指定的JSON格式。
    """
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

        valid_updated_scores = {}

        for topic, score_data in updated_scores.items():
            if isinstance(score_data, dict):
                score = score_data.get("score")
                reason = score_data.get("reason", "")  
                if isinstance(score, int) and 0 <= score <= 4:
                    valid_updated_scores[topic] = {"score": score, "reason": reason}
                else:
                    print(f"无效的更新分数：主题='{topic}', 分数='{score}'")
            else:
                print(f"无效的 updated_scores 格式：主题='{topic}', 数据='{score_data}'")

        return summary, valid_updated_scores
    except json.JSONDecodeError:
        print("无法解析 SummaryAgent 的输出，请确保其遵循JSON格式。")
        return "", {}



def generate_score_table(memory, symptom_level):
    total_score = sum(item.get("updated_score", item["score"]) for item in memory)
    df = pd.DataFrame({
        "序号": range(1, len(memory) + 1),
        "项目": [item["topic"] for item in memory],
        "得分": [item["score"] for item in memory],
        "更新后得分": [item.get("updated_score", item["score"]) for item in memory]
    })
    df.loc[len(df.index)] = ["", "总得分", total_score, ""]  
    df.loc[len(df.index)] = ["", "症状等级", symptom_level, ""]  
    return df.to_markdown(index=False)


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
        logger.info("必要性评分 == 2，继续深入提问。")
        dialog_print("必要性评分 == 2，继续深入提问。")
        return True
    elif necessity_score == 1:
        if asked_questions < 2:
            logger.info("必要性评分 == 1，且提问次数未达到2次，继续提问。")
            dialog_print("必要性评分 == 1，且提问次数未达到2次，继续提问。")
            return True
        else:
            logger.info("必要性评分 == 1，且提问次数已达到2次，停止提问。")
            dialog_print("必要性评分 == 1，且提问次数已达到2次，停止提问。")
            return False
    else:
        logger.info("必要性评分 == 0，停止提问。")
        dialog_print("必要性评分 == 0，停止提问。")
        return False
    

def generate_mock_response(question, topic, identification, real_interview, hamd_17_scores, scoring_standard=None, current_topic_history=None, depth=0, clear_memory=False):

    if clear_memory:
        prompt = "请忘记之前的所有对话内容，重新开始新的会话。"
        logger.info("正在清除对话记忆...")
    else:
        interview_history = ""
        for para in real_interview:
            role = para.get("roleName", "未知角色")
            content = para.get("content", "")
            interview_history += f"{role}: {content}\n"

        hamd_scores = hamd_17_scores.get("items", {})
        hamd_scores_str = "\n".join([f"{k}: {v}" for k, v in hamd_scores.items()])


        if identification == "":
            prompt = (
                f"你是一位正在参与心理评估的用户。以下是之前的真实医生-患者访谈对话以及你的汉密尔顿抑郁量表HAMD-17量表真实得分信息。\n\n"
                f"### 真实访谈对话：\n{interview_history}\n"
                f"### HAMD-17量表得分：\n{hamd_scores_str}\n\n"
                f"请基于以上信息，回答以下问题：\n"
                f"冰糖提问：{question}\n\n"
                f"请提供您的基本信息（年龄、性别、职业），您的回答（例如：25，男，工程师 或 年龄:25, 性别:男, 职业:工程师）："
            )
        else:
            if depth == 0:
                prompt = (
                    f"你是一位正在参与心理评估的用户。以下是你的汉密尔顿抑郁量表HAMD-17量表真实得分信息。\n\n"
                    f"### HAMD-17量表得分：\n{hamd_scores_str}\n\n"
                    f"基本信息：{identification}\n"
                    f"冰糖提问：{question}\n"
                    f"请根据你的真实量表的各主题得分信息，给出真实且合理的回答，回答不超过50个字："
                )
            else:
                scoring_standard_str = "\n".join([f"{k} - {v}" for k, v in scoring_standard.items()])

                topic_history_str = ""
                if current_topic_history:
                    for qa in current_topic_history:
                        topic_history_str += f"冰糖提问：{qa['question']}\n用户回答：{qa['response']}\n"

                prompt = (
                    f"你是一位正在参与心理评估的用户，正在接受心理评估访谈。\n"
                    f"以下是你的汉密尔顿抑郁量表HAMD-17量表真实得分信息：\n{hamd_scores_str}\n\n"
                    f"基本信息：{identification}\n\n"
                    f"**当前主题已进行的对话：**\n{topic_history_str}\n"
                    f"冰糖深入提问：{question}\n\n"
                    f"**当前主题：{topic} 的评分标准如下：**\n{scoring_standard_str}\n\n"
                    f"请基于你之前给出的所有回答、真实量表得分，**严格参考主题评分标准**，针对当前问题给出符合标准的真实且合理的回答，禁止虚构症状情况进行回答。\n"
                    f"当问题内容超出量表信息范围无法准确作答时，可选择回答“不清楚”或类似表达，使用时需委婉表达并给出符合患者信息的理由，但请尽量避免频繁使用这类回答。"
                )

        logger.info(f"生成模拟回复的提示：{prompt}")

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:10010/v1/",
            api_key="none"
        )
        completion = client.chat.completions.create(
            model="deepseek-r1-32b",
            messages=[
                {"role": "system", "content": "你是一位正在参与心理评估的用户的模拟者。"},  
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



def save_assessment_results(video_name, overall_score, symptom_level, updated_scores, csv_file="depression.csv"):
    """
    将评估结果保存为CSV文件。如果video_name已存在，则覆写该行，否则追加新行。

    :param video_name: 视频名称
    :param overall_score: 总评分
    :param symptom_level: 症状等级
    :param updated_scores: 各主题的更新分数字典
    :param csv_file: 保存的CSV文件名
    """
    data = {
        "video_name": video_name,
        "total": overall_score,
        "classes": symptom_level
    }

    score_values = {topic: details["score"] for topic, details in updated_scores.items()}

    for idx, (topic, score) in enumerate(score_values.items(), 1):
        item_key = f"item{idx}"
        data[item_key] = score

    for i in range(1, 18):
        item_key = f"item{i}"
        if item_key not in data:
            data[item_key] = 0  

    if os.path.isfile(csv_file):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"读取CSV文件 {csv_file} 时发生错误：{e}")
            print(f"读取CSV文件时发生错误，请检查日志。")
            return

        if video_name in df['video_name'].values:
            index = df.index[df['video_name'] == video_name].tolist()[0]

            for key, value in data.items():
                df.at[index, key] = value
            logger.info(f"已更新视频 {video_name} 的评估结果。")
        else:
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            logger.info(f"已追加视频 {video_name} 的评估结果。")
    else:
        columns = ["video_name", "total", "classes"] + [f"item{i}" for i in range(1, 18)]
        df = pd.DataFrame([data], columns=columns)
        logger.info(f"已创建新的CSV文件 {csv_file} 并保存评估结果。")

    try:
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"评估结果已保存到 {csv_file}")
    except Exception as e:
        logger.error(f"保存CSV文件 {csv_file} 时发生错误：{e}")
        print(f"保存CSV文件时发生错误，请检查日志。")


def is_file_already_evaluated(video_name, csv_file_path):
    """
    检查CSV文件中是否已经存在指定video_name的记录。
    """
    if not os.path.isfile(csv_file_path):
        return False 

    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        return video_name in df['video_name'].values
    except Exception as e:
        logger.error(f"检查CSV文件 {csv_file_path} 时发生错误：{e}")
        return False 

