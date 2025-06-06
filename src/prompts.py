import json

def build_question_prompt(topic, last_question, last_response, identification, chatprompt, memory, all_topics, depth=0):
    # 格式化整个memory中的各条目信息
    memory_entries = "\n".join([
        f"主题: {entry['topic']}    评分: {entry['score']} 分   得分依据: {entry['summary']}\n"
        for entry in memory
    ]) if memory else "无"

    other_topics = [t for t in all_topics if t != topic]
    other_topics_str = ", ".join(other_topics)


    if depth == 0:
        # 初始提问
        _prompt = f"""你是专业的心理咨询助手冰糖，具备高度的共情能力，能够与用户进行深入沟通。
上一个问题：{last_question}
用户的回答：{last_response}

首先，请对用户的回答给予一定的真实有根据、具体且真诚的反馈和共情回应，避免添加任何用户未提及的信息。

接下来，针对主题'{topic}'进行访谈，请提出一个相关的问题。
请确保新问题不涉及以下已讨论的信息：
{memory_entries}

特别注意：
1. 问题内容应符合用户的身份信息：{identification}。
2. 仅提供问题的文字内容，不要添加其他任何无关内容。
3. 每个问题不超过150字。

示例问题：
- {chatprompt[topic][0]}
- {chatprompt[topic][1]}
"""
    else:
        # 深入提问
        _prompt = f"""你是专业的心理咨询助手冰糖，具备高度的共情能力，能够与用户进行深入沟通。
上一个问题：{last_question}
用户的回答：{last_response}

首先，请针对用户的回答给予一定的真实有根据的、具体且真诚的反馈和共情回应，避免添加任何用户未提及的信息。
接下来，针对主题'{topic}'进行深入探讨，请提出一个更具洞察力更有临床价值且简单易答的问题。
请结合用户的回答，灵活选择提问方式。可以提出选择型问题，也可以提出开放式问题，但都要围绕症状的程度、频率、持续时间、影响范围等方面展开。

**请注意以下几点：**
- **选择型问题：** 适用于需要明确程度或频率的情况，例如“您觉得这种感觉是（轻微、中等、严重）中的哪一种？”
- **开放式问题：** 适用于需要用户更详细描述情况的情况，例如“这种症状对您的生活造成了哪些具体影响？”
- **避免重复：** 避免重复提问类似的问题。

特别注意：
1. 问题内容应符合用户的身份信息：{identification}。
2. 避免提出与已讨论信息重复的问题。
已讨论信息：
{memory_entries}

3. 确保提出的问题不涉及到其他主题：
{other_topics_str}

4. 提出的问题需易于回答，帮助用户更准确地描述情况。
5. 仅提供问题的文字内容，不要添加其他任何无关内容。
6. 每个问题不超过150字。
"""

    return _prompt


def build_necessity_prompt(topic, current_topic_history):
    """
    构建必要性评分的提示
    """
    topic_history_str = ""
    for qa in current_topic_history:
        topic_history_str += f"冰糖提问：{qa['question']}\n用户回答：{qa['response']}\n"

    necessity_prompt = (
        f"以下是用户在主题'{topic}'下的所有问答内容，请根据这些内容严格评估是否需要进一步提问。\n"
        f"{topic_history_str}\n"
        "请按照以下标准进行必要性评分（0-2分）：\n"
        "【评分标准】\n"
        "0 - 不需要进一步深入探讨，当前信息已足够。\n"
        "1 - 有一定必要进一步深入探讨，可能存在潜在信息。\n"
        "2 - 非常有必要进一步深入探讨，用户的心理状态需要更深入的了解。\n"
        "【评估维度】\n"
        "1. 信息充分性：用户回答的信息越简略，进一步提问的必要性越高。\n"
        "2. 症状严重性：用户回答中表现出的症状越严重，进一步提问的必要性越高。\n"
        "【输出要求】\n"
        "1. 仅返回数字评分（0/1/2），禁止任何解释性文字\n"
        "2. 当出现明显否定回答时，如‘没有’等，则无需进一步提问。\n\n"
    )

    return necessity_prompt


def build_scoring_prompt(topic, current_topic_history, scale_name, scoring_standard):
    """
    构建评分的提示，根据不同的量表切换评分标准。

    :param topic: 当前评估的主题
    :param current_topic_history: 当前主题下的问答历史
    :param scale_name: 当前使用的量表名称（例如 "HAMD-17"）
    :param scoring_standard: 当前量表的评分标准字典
    :return: 评分提示字符串
    """
    topic_history_str = ""
    for qa in current_topic_history:
        topic_history_str += f"提问：{qa['question']}\n用户的回答：{qa['response']}\n"

    # 获取当前主题的评分标准
    if topic in scoring_standard:
        theme_scoring = scoring_standard[topic]
    else:
        theme_scoring = {
            "0": "无症状。",
            "1": "轻度症状。",
            "2": "中度症状。",
            "3": "较重症状。",
            "4": "重度症状。"
        }

    # 构建评分标准字符串
    scoring_standard_str = "\n".join([f"{k} - {v}" for k, v in theme_scoring.items()])

    score_prompt = (
        f"请根据以下内容对用户在主题'{topic}'下的回答进行评分（0-4分）：\n"
        f"{topic_history_str}\n\n"
        "评分标准：\n"
        f"{scoring_standard_str}\n\n"
        "特别注意：评分时请专注于心理相关内容，并遵循评分标准进行评分以及不要超过评分标准的限制。\n"
        "此外，请总结用户的回答内容，并提供一到两句话作为评分依据。\n\n"
        "请严格按照以下JSON格式输出，不包含其他任何内容：\n"
        "{\n"
        '    "score": <评分>,\n'
        '    "summary": "<总结>"\n'
        "}"
    )
    return score_prompt


def build_summary_prompt(total_history_str, scores_str, memory):
    """
    构建总结和建议的提示
    """
    summary_prompt = (
        "你是一位专业的心理咨询助手，请根据以下内容生成详细的总结和建议，即用户的主述症状摘要和个性化针对性的建议：\n"
        "1. 完整的对话历史\n"
        "2. 各主题的评分结果\n"
        "3. 当前的记忆内容（memory）\n\n"
        "要求：\n"
        "1. 使用正式且专业的语言撰写总结与建议。\n"
        "2.  仔细分析对话历史和记忆内容，对各主题的初始评分进行合理的细微调整，以更准确地反映用户的心理状态，更符合心理患者症状的分布。调整应提供简要的理由，避免随意更改。\n\n"
        f"对话历史和得分：\n{total_history_str}\n\n得分：{scores_str}\n\n当前 memory 内容：\n{json.dumps(memory, ensure_ascii=False, indent=2)}\n\n"
        "请严格按照以下JSON格式输出，不包含其他任何内容：\n\n"
        "{\n"
        '    "summary": "<总结与建议>",\n'
        '    "updated_scores": {\n'
        '        "<主题1>": {"score": <更新后分数1>, "reason": "<调整理由1>"}, \n'
        '        "<主题2>": {"score": <更新后分数2>, "reason": "<调整理由2>"}, \n'
        '        "..." : {"score": "...", "reason": "..."} \n'
        '    }\n'
        "}"
    )
    return summary_prompt



def build_report_prompt(report_table, summary, symptom_level):
    """
    构建报告的提示
    """
    report_prompt = (
        "你是一位专业的心理咨询助手，请根据以下输入生成一份格式化的心理评估报告。\n\n"
        "输入内容如下：\n\n"
        "### 得分表\n"
        f"{report_table}\n\n"
        "### 总结与建议\n"
        f"{summary}\n\n"
        "要求：\n"
        "1. **报告结构**：\n"
        "   - **报告标题**：例如“心理评估报告”。\n"
        "   - **用户得分表**：根据提供的得分表数据展示。\n"
        "   - **总结与建议**：直接使用提供的总结与建议内容。\n\n"
        "2. **内容处理**：\n"
        "   - 不要对输入的得分表和总结与建议内容进行任何修改。\n"
        "3. **语言风格**：使用清晰且专业的语言，适合直接呈现给用户。\n\n"
        "请严格按照上述要求生成心理评估报告，不添加其他任何内容。"
    )
    return report_prompt