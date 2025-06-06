import json
import os
from openai import OpenAI
from statistics import mean


api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("未找到API密钥，请设置环境变量 OPENAI_API_KEY")


client = OpenAI(base_url="https://xiaoai.plus/v1", api_key=api_key)



evaluation_criteria = """
采用以下4个指标对心理健康评估过程进行评估（评分范围为0-1，允许0.05的增量）：

1. 对话连贯性（Dialogue Coherence）：评估心理评估过程是否在语法和语义上与上下文连贯。例如，回复是否与问题相关，是否避免了答非所问的情况。
   - 0：完全脱节，回答与问题完全无关，或存在大量语法错误导致无法理解
   - 0.2：严重不连贯，大部分回答偏离上下文，频繁出现答非所问
   - 0.4：部分不连贯，约50%的回答与上下文逻辑断裂，存在明显跳跃或重复
   - 0.6：基本连贯，70%的回答与上下文相关，但偶尔出现逻辑跳跃或冗余信息
   - 0.8：高度连贯，90%的回答逻辑流畅，仅个别语句不够自然
   - 1.0：完美连贯，所有回答与上下文无缝衔接，语法语义完全一致

2. 用户满意度（User Satisfaction）：评估用户对心理健康评估系统的整体满意度，包括问答的准确性、连贯性和情感支持。
   - 0：完全无法接受，回答错误或冒犯用户，导致对话终止
   - 0.2：非常不满意，回答明显错误或缺乏情感支持，用户体验极差
   - 0.4：部分不满意，约50%的回答存在准确性或情感支持不足问题
   - 0.6：基本满意，回答基本准确但缺乏深度，情感支持较为机械
   - 0.8：高度满意，回答准确且情感自然，能有效缓解用户焦虑
   - 1.0：完美体验，回答精准、个性化，显著提升用户心理安全感

3. 可解释性（Interpretability）：评估系统是否能够提供清晰的推理路径或证据支持其结果。例如，心理助手的评分、报告等是否基于明确的逻辑或证据链。
   - 0：毫无逻辑，缺乏任何推理路径或证据支持
   - 0.2：逻辑混乱，推理路径模糊，证据支持不足
   - 0.4：部分可解释，30%包含简单推理，但证据链不完整
   - 0.6：基本可解释，50%有明确逻辑，但存在跳跃性推理
   - 0.8：高度可解释，80%包含完整证据链，符合心理学评估标准
   - 1.0：完美可解释，所有结论均有清晰的逻辑路径和临床依据支持

4. 任务完成度（Task Completion）：评估系统是否成功完成了预设任务。例如，心理助手是否完成了HAMD-17量表的评估，并提供了合理的报告和建议。
   - 0：系统完全未完成预设任务，评估和建议均不合理
   - 0.2：系统部分完成预设任务，但评估和建议存在重大遗漏
   - 0.4：系统基本完成预设任务，但评估和建议存在一定不合理之处
   - 0.6：系统较好完成预设任务，报告内容基本可用但缺乏细节
   - 0.8：系统高度完成预设任务，报告结构完整且建议合理
   - 1.0：超额完成任务（如发现潜在风险并提供转诊建议），报告专业性强
"""


json_file_path = "baseline_10.json"
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # qa_dialogue = data['qa_dialogue']
except FileNotFoundError:
    print(f"错误：未找到文件 {json_file_path}，请检查路径是否正确")
    exit(1)


all_scores = {
    "dialogue_coherence": [],
    "user_satisfaction": [],
    "interpretability": [],
    "task_completion": []
}



for idx, evaluation in enumerate(data, 1):
    try:
        qa_dialogue = evaluation['qa_dialogue']
    except KeyError:
        print(f"错误：第 {idx} 个测评缺少 'qa_dialogue' 键，跳过此测评")
        continue




    dialogue_text = ""
    for turn in qa_dialogue:
        question = turn.get('question', '无问题')  # 使用get方法防止键缺失
        answer = turn.get('answer', '无回答')
        score = turn.get('score', '无评分')  # score可能是可选字段
        dialogue_text += f"心理助手: {question}\n用户: {answer}\n"
        if score != '无评分':
            dialogue_text += f"评分: {score}\n"

    if "report" in evaluation:
        dialogue_text += evaluation["report"] + "\n"


    prompt = f"""
    你是一个专业的心理健康评估系统分析师，需要严格按照评分标准进行量化评估。
    请评估以下心理评估对话过程，并根据提供的标准给出四个指标的评分：对话连贯性、用户满意度、可解释性和任务完成度，并给出简要的评分理由。每个指标的评分范围是0到1，允许0.05的增量。
    
    **心理评估过程：**
    {dialogue_text}
    
    **评估标准：**
    {evaluation_criteria}
    
    请以JSON格式返回评估结果，例如：
    {{
      "dialogue_coherence": {{
        "score": 0.8,
        "reason": "对话逻辑流畅，90%以上的回答与上下文相关，仅有少量冗余。"
      }},
      "user_satisfaction": {{
        "score": 0.8,
        "reason": "回答准确且情感自然，能缓解焦虑，但情感支持深度可提升。"
      }},
      "interpretability": {{
        "score": 0.8,
        "reason": "80%以上的结论有完整证据链，符合评估标准，但评分依据说明不足。"
      }},
      "task_completion": {{
        "score": 0.8,
        "reason": "高度完成HAMD-17评估，报告结构完整且建议合理，但未识别潜在风险。"
      }}
    }}
    """


    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",  
            messages=[
                {"role": "system", "content": "你是一个专业的心理健康评估系统分析师，能够根据评分标准对系统评估过程给出客观的评分。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0, 
            max_tokens=2048  
        )


        evaluation_result = response.choices[0].message.content

        # print(prompt)

        try:
            if evaluation_result.startswith("```json"):
                evaluation_result = evaluation_result[len("```json"):].strip()
            elif evaluation_result.startswith("```"):
                evaluation_result = evaluation_result[len("```"):].strip()

            if evaluation_result.endswith("```"):
                evaluation_result = evaluation_result[:-len("```")].strip()


            scores = json.loads(evaluation_result)
            print(f"\n### 第 {idx} 个测评（{evaluation['json_file']}）评分结果：")
            for key, value in scores.items():
                print(f"- **{key}**: {value['score']}")
                print(f"  - 理由: {value['reason']}")
                all_scores[key].append(value['score'])
        except json.JSONDecodeError:
            print(f"错误：第 {idx} 个测评无法解析大模型返回的评估结果，以下是原始输出：")
            print(evaluation_result)
    except Exception as e:
        print(f"错误：第 {idx} 个测评API调用失败，详细信息如下：")
        print(str(e))


average_scores = {key: mean(scores) for key, scores in all_scores.items()}

print("\n### 所有10个测评的平均评分：")
for key, value in average_scores.items():
    print(f"- **{key}**: {value:.2f}")