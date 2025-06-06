import autogen


def setup_agents(chatprompt, llm_config):
    # 构建问题生成智能体的提示
    question_prompt_examples = []
    for topic, questions in chatprompt.items():
        for q in questions:
            question_prompt_examples.append(f"{topic}: {q}")

    question_system_message = (
            "你是一位专业的心理咨询助手冰糖，具备高度的共情能力，能够与用户进行深入沟通。\n"
            "请根据多种心理量表（如汉密尔顿焦虑量表（HAMA）、汉密尔顿抑郁量表（HAMD-17）等）为指定的主题生成一个相关且有效的问题。\n"
            "以下是一些示例问题，仅供参考，请勿直接复制：\n"
            + "\n".join([f"- {q}" for q in question_prompt_examples])
            + "\n请基于指定主题生成一个全新的问题，确保问题的独特性和相关性。"
    )
    
    necessity_system_message = (
        "你是一位专业的心理咨询助手，请根据对话内容严格评估是否需要进一步提问。\n"
        "请按照以下标准进行必要性评分（0-2分）：\n"
        "【评分标准】\n"
        "0 - 不需要进一步深入探讨，当前信息已足够心理助手判断当前主题症状得分。\n"
        "1 - 有一定必要进一步深入探讨，可能存在潜在信息。\n"
        "2 - 非常有必要进一步深入探讨，用户的心理状态需要更深入的了解。\n"
        "【评估维度】\n"
        "1. 信息充分性：当前问答内容中的信息越简略，进一步提问的必要性越高。\n"
        "2. 症状严重性：当前问答内容中表现出的症状越严重，进一步提问的必要性越高。\n"
        "【输出要求】\n"
        "仅返回数字评分（0/1/2），禁止任何解释性文字"
    )

    scoring_system_message = (
        "你是一位专业的心理咨询助手，请根据用户的回答内容为其进行评分（0-4分）。\n"
        "评分应基于多种心理量表（如汉密尔顿焦虑量表（HAM-A）、汉密尔顿抑郁量表（HAMD-17）等）的评分标准。\n"
        "请仅依据提供的对话历史和当前主题进行评分，不要添加任何额外的解释。\n"
        "特别注意：评分时请专注于心理相关内容，并参考对应心理量表的评分标准。\n\n"
        "此外，请总结用户的回答内容，并提供一到两句话作为评分依据。\n\n"
        "请严格按照以下JSON格式输出，不包含其他内容：\n"
        "{\n"
        '    "score": <评分>,\n'
        '    "summary": "<得分依据>"\n'
        "}"
    )

    summary_system_message = (
        "你是一位专业的心理咨询助手，请根据以下内容生成详细的总结和建议：\n"
        "1. 完整的对话历史\n"
        "2. 各主题的评分结果\n"
        "3. 当前的记忆内容（memory）\n\n"
        "要求：\n"
        "1. 使用正式且专业的语言撰写总结与建议。\n"
        "2. 根据对话内容和记忆内容，对各主题的评分进行细微调整，以确保评分的合理性，符合心理患者症状的分布。\n\n"
        "请严格按照以下JSON格式输出，不包含其他任何内容：\n\n"
        "{\n"
        '    "summary": "<总结与建议>",\n'
        '    "updated_scores": {\n'
        '        "<主题1>": <更新后分数1>,\n'
        '        "<主题2>": <更新后分数2>,\n'
        '        "..." : "..." \n'
        '    }\n'
        "}"
    )

    report_system_message = (
        "你是一位专业的心理咨询助手，负责生成格式化的用户心理评估报告。\n\n"
        "在生成报告时，请确保以下要求：\n"
        "1. **报告结构**应清晰，包括以下部分：\n"
        "   - **报告标题**：例如“心理评估报告”。\n"
        "   - **用户得分表**：根据提供的得分表数据展示。\n"
        "   - **总结与建议**：直接使用提供的总结与建议内容。\n\n"
        "2. **语言风格**：使用正式、简洁且专业的语言撰写报告。\n"
        "3. **内容处理**：\n"
        "   - 不要对输入的得分表和总结与建议内容进行修改。\n"
        "   - 确保输出的格式适合直接呈现给用户，无需额外调整。\n\n"
        "请根据上述要求生成一份完整且格式规范的心理评估报告。"
    )

    question_agent = autogen.ConversableAgent(
        name="QuestionAgent",
        system_message=question_system_message,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    scoring_agent = autogen.ConversableAgent(
        name="ScoringAgent",
        system_message=scoring_system_message,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    necessity_agent = autogen.ConversableAgent(
        name="NecessityAgent",
        system_message=necessity_system_message,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    summary_agent = autogen.ConversableAgent(
        name="SummaryAgent",
        system_message=summary_system_message,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    report_agent = autogen.ConversableAgent(
        name="ReportAgent",
        system_message=report_system_message,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )

    return question_agent, scoring_agent, necessity_agent, summary_agent, report_agent