import json
import os
from openai import OpenAI
from statistics import mean
import csv


api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("未找到API密钥，请设置环境变量 OPENAI_API_KEY")

client = OpenAI(base_url="https://xiaoai.plus/v1", api_key=api_key)


with open('HAMD-17.json', 'r', encoding='utf-8') as f:
    hamd_17 = json.load(f)

with open('scoring_standards.json', 'r', encoding='utf-8') as f:
    scoring_standards = json.load(f)

realdata = []
with open('realdata_10.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        realdata.append(row)

json_file_path = "agent72-14b.json"

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # # 提取qa_dialogue部分
        # qa_dialogue = data['qa_dialogue']
except FileNotFoundError:
    print(f"错误：未找到文件 {json_file_path}，请检查路径是否正确")
    exit(1)


all_scores_question = {
    "clinical_relevance": [],
    "discriminant_validity": []
}

all_scores_answer = {
    "accuracy": [],
    "hallucination_rate": [],
    "consistency": []
}


for idx, (evaluation, real_patient_data) in enumerate(zip(data, realdata), 1):
    print("===="*25)

    try:
        json_file = evaluation['json_file']
    except KeyError:
        print(f"错误：第 {idx} 个测评缺少 'json_file' 键，跳过此测评")
        continue

    dialogue_text_question = ""
    dialogue_text_answer = ""
    for key, value in evaluation.items():
        if key not in ["json_file", "report"] and isinstance(value, dict):
            section_title = f"--- 评估主题：{key} ---\n"
            if "qa_dialogue" in value:
                for turn in value["qa_dialogue"]:
                    question = turn.get('question', '无问题')
                    answer = turn.get('answer', '无回答')
                    dialogue_text_question += f"问题: {question}\n"
                    dialogue_text_answer += f"问题: {question}\n模拟回答: {answer}\n"

    if not dialogue_text_question or not dialogue_text_answer:
        print(f"错误：第 {idx} 个测评（{json_file}）没有找到任何对话内容，跳过此测评")
        continue

    real_scores = [real_patient_data[f'item{i+1}'] for i in range(17)]
    themes = list(scoring_standards.keys())


    evaluation_criteria_question = f"""
    根据汉密尔顿抑郁量表HAMD-17内容以及HAMD-17量表的评分标准，采用以下2个指标对心理健康评估过程中的问题进行评估（评分范围为0-1，允许0.01的增量）：
    
    1. 临床相关性（Clinical Relevance）：问题是否能够精准覆盖量表所需的核心症状信息，评估问题内容与标准化的心理健康评估工具（如HAMD-17）的一致性。
       - 0.2：问题内容与核心症状几乎无关，仅能涉及少部分症状。
       - 0.4：问题覆盖较少核心症状，且存在较大偏差。
       - 0.6：问题涵盖症状的部分核心内容，但表述较为模糊，或未能精准覆盖。
       - 0.8：问题涵盖大部分核心症状，但存在一些较小的差异或表述不同。
       - 1.0：问题准确无误地涵盖核心症状，问题内容与标准化工具完全一致。
    
    2. 区分效度（Discriminant Validity）：问题是否能够有效区分不同症状的严重程度，确保回答能够准确映射到量表的评分层级。
       - 0.2：问题无法有效区分不同的症状严重程度，评分存在较大误差。
       - 0.4：问题对于症状的严重度区分较弱，影响量表的有效评分。
       - 0.6：问题能区分部分症状严重性，但无法精准映射评分。
       - 0.8：问题在一定程度上能够区分症状，但可能在某些情况下难以分辨不同级别。
       - 1.0：问题能够清晰区分不同症状的严重程度，并与量表评分一致。
    
    汉密尔顿抑郁量表HAMD-17包含的主题项：
    {list(hamd_17.keys())}
    
    HAMD-17量表的各主题评分标准：
    {json.dumps(scoring_standards, ensure_ascii=False, indent=2)}
    """

    prompt_question = f"""
    你是一个专业的心理健康系统分析师，需要严格按照评分标准进行量化评估。
    请评估以下心理评估对话过程中的问题，并根据提供的评估标准给出两个指标的评分：临床相关性和区分效度，并给出简要的评分理由。每个指标的评分范围是0到1，允许0.01的增量。
    
    **心理评估对话过程中的问题内容：**
    {dialogue_text_question}
    
    **评估标准：**
    {evaluation_criteria_question}
    
    **特别注意：**
    请以JSON格式返回评估结果，不要添加任何多余内容，例如：
    {{
      "clinical_relevance": {{
        "score": 0.8,
        "reason": "问题内容与核心症状高度相关，仅有轻微表述差异。"
      }},
      "discriminant_validity": {{
        "score": 0.8,
        "reason": "问题能够清晰区分不同症状的严重程度，偶尔有轻微误差。"
      }}
    }}
    """


    evaluation_criteria_answer = f"""
    根据患者的汉密尔顿抑郁量表HAMD-17量表真实得分信息以及HAMD-17量表的评分标准，采用以下3个指标对心理健康评估过程中的模拟回答进行评估（评分范围为0-1，允许0.01的增量）：

    1. 准确性（Accuracy）：回答是否符合真实的得分与评分标准。
       - 0.2：回答完全不符合评分标准，存在重大误差。
       - 0.4：回答偏离评分标准较大，难以与标准对齐。
       - 0.6：回答与评分标准有所出入，但整体符合情境。
       - 0.8：回答基本符合评分标准，但存在少许偏差。
       - 1.0：回答完全符合评分标准和量表要求。

    2. 幻觉率（Hallucination Rate）：统计大模型模拟回答中与真实患者量表得分不一致的比例。
       - 0.2：幻觉严重，几乎所有回答都与标准不一致。
       - 0.4：幻觉较为严重，影响评分结果。
       - 0.6：幻觉出现频率较高，但总体影响较小。
       - 0.8：幻觉较少，偶尔出现不一致的回答。
       - 1.0：无幻觉，所有回答与得分标准一致。

    3. 一致性（Consistency）：回答是否在不同主题和情境下逻辑一致，保持连贯性。
       - 0.2：回答存在严重的逻辑不一致，影响整体评估结果。
       - 0.4：回答逻辑不一致，导致评估内容的可靠性下降。
       - 0.6：回答有部分不一致的地方，且表述上有所矛盾。
       - 0.8：回答大体一致，但有轻微逻辑偏差或偶尔的表述不清。
       - 1.0：回答在不同情境下逻辑一致，表述清晰，前后无矛盾。
    
    HAMD-17量表真实得分信息：
    {real_scores}
    
    HAMD-17量表的评分标准：
    {json.dumps(scoring_standards, ensure_ascii=False, indent=2)}
    """

    prompt_answer = f"""
    你是一个专业的心理健康评估系统分析师，需要严格按照评分标准进行量化评估。
    请评估以下心理评估对话过程中的模拟回答，并根据提供的评估标准给出三个指标的评分：准确性、幻觉率和一致性，并给出简要的评分理由。每个指标的评分范围是0到1，允许0.01的增量。
    
    **心理评估对话过程中的模拟回答内容：**
    {dialogue_text_answer}
    
    **评估标准：**
    {evaluation_criteria_answer}

    **特别注意：**
    请以JSON格式返回评估结果，不要添加任何多余内容，例如：
    {{
      "accuracy": {{
        "score": 0.8,
        "reason": "回答符合评分标准，仅有轻微偏差。"
      }},
      "hallucination_rate": {{
        "score": 0.8,
        "reason": "大部分回答与真实得分一致，仅有个别不一致。"
      }},
      "consistency": {{
        "score": 0.8,
        "reason": "回答在大多数情境下逻辑一致，仅有个别矛盾。"
      }}
    }}
    """

    try:
        response_question = client.chat.completions.create(
            model="gpt-4o-2024-11-20",  
            messages=[
                {"role": "system",
                 "content": "你是一个专业的心理健康评估系统分析师，能够根据评分标准给出客观的评分。"},
                {"role": "user", "content": prompt_question},
            ],
            temperature=0,  
            max_tokens=4096  
        )
        response_answer = client.chat.completions.create(
            model="gpt-4o-2024-11-20",  
            messages=[
                {"role": "system",
                 "content": "你是一个专业的心理健康评估系统分析师，能够根据评分标准给出客观的评分。"},
                {"role": "user", "content": prompt_answer},
            ],
            temperature=0,  
            max_tokens=4096  
        )

        evaluation_result_question = response_question.choices[0].message.content
        evaluation_result_answer = response_answer.choices[0].message.content


        try:
            if evaluation_result_question.startswith("```json"):
                evaluation_result_question = evaluation_result_question[len("```json"):].strip()
            elif evaluation_result_question.startswith("```"):
                evaluation_result_question = evaluation_result_question[len("```"):].strip()
            if evaluation_result_question.endswith("```"):
                evaluation_result_question = evaluation_result_question[:-len("```")].strip()
            scores_question = json.loads(evaluation_result_question)
            
            # scores_question = json.loads(evaluation_result_question.strip("```json").strip("```").strip())
            print(f"\n### 第 {idx} 个测评（{evaluation['json_file']}）问题评分结果：")
            for key, value in scores_question.items():
                print(f"- **{key}**: {value['score']}")
                print(f"  - 理由: {value['reason']}")
                all_scores_question[key].append(value['score'])
            print("----"*25)

            if evaluation_result_answer.startswith("```json"):
                evaluation_result_answer = evaluation_result_answer[len("```json"):].strip()
            elif evaluation_result_answer.startswith("```"):
                evaluation_result_answer = evaluation_result_answer[len("```"):].strip()
            if evaluation_result_answer.endswith("```"):
                evaluation_result_answer = evaluation_result_answer[:-len("```")].strip()
            scores_answer = json.loads(evaluation_result_answer)
            # scores_answer = json.loads(evaluation_result_answer.strip("```json").strip("```").strip())
            print(f"\n### 第 {idx} 个测评（{evaluation['json_file']}）回答评分结果：")
            for key, value in scores_answer.items():
                print(f"- **{key}**: {value['score']}")
                print(f"  - 理由: {value['reason']}")
                all_scores_answer[key].append(value['score'])
        except json.JSONDecodeError:
            print(f"错误：第 {idx} 个测评无法解析大模型返回的评估结果，以下是原始输出：")
            print(evaluation_result_question)
            print(evaluation_result_answer)
    except Exception as e:
        print(f"错误：第 {idx} 个测评API调用失败，详细信息如下：")
        print(str(e))


average_scores_question = {key: mean(scores) for key, scores in all_scores_question.items()}
average_scores_answer = {key: mean(scores) for key, scores in all_scores_answer.items()}


print("\n### 所有测评的问题平均评分：")
for key, value in average_scores_question.items():
    print(f"- **{key}**: {value:.2f}")

print("\n### 所有测评的回答平均评分：")
for key, value in average_scores_answer.items():
    print(f"- **{key}**: {value:.2f}")