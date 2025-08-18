import pandas as pd
import logging
import autogen
import json
import os
from utils import get_valid_input, categorize_score, extract_score_and_summary, extract_summary_and_updated_scores, generate_score_table, is_necessary, makerequest, generate_mock_response, parse_personal_info, extract_score, save_assessment_results, is_file_already_evaluated, custom_speaker_selection_func, generate_report
from agents import setup_agents
from data_load import load_real_data
from logging_setup import dialog_print
from config import get_llm_config
from memory import MemoryGraph


llm_config = get_llm_config()


logger = logging.getLogger(__name__)


def perform_assessment(topics, chatprompt, agents, scale_name, scoring_standards, real_interview, scale_scores, automated=False):
    try:
        logger.info("开始执行心理评估任务。")
        question_agent, scoring_agent, necessity_agent, summary_agent, user_proxy = agents
        groupchat = autogen.GroupChat(
            agents=[question_agent, scoring_agent, necessity_agent, summary_agent, user_proxy],
            messages=[],
            max_round=2,
            speaker_selection_method=custom_speaker_selection_func,
        )
        group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        total_history = []
        scores = []
        identification = ""
        last_topic = "基本信息收集"
        last_question = ""
        last_response = ""
        qa_count = 0


        groupchat.reset()  

        initial_message = (
            f"Hello, I am your dedicated psychological assistant. I will conduct an interview with you based on {scale_name} to assess the severity of related symptoms. Please note that this is only a preliminary screening and cannot replace formal psychiatric diagnosis and treatment. "
            "First, for the accuracy of the assessment, I would like to collect your basic information: age, gender, occupation. If you're ready, let's begin."
        )
        dialog_print(f"冰糖提问: {initial_message}")
        logger.info(f"发送初始消息：{initial_message}")
        logger.info("初始消息已发送给用户。")

        if automated:
            user_response = generate_mock_response(initial_message, topic=None, identification="", real_interview=real_interview, scale_scores=scale_scores, 
                                                   scoring_standard=None, current_topic_history=None, scale_name=scale_name)
            dialog_print(f"模拟回答：{user_response}")
            age, gender, occupation = parse_personal_info(user_response)
            if age is None or gender is None or occupation is None:
                logger.error("自动化模式下，无法正确解析基本信息。")
                return "自动化模式下，无法正确解析基本信息。"
        else:
            while True:
                user_response = get_valid_input("Your response (e.g., 25, male, engineer or age:25, gender:male, occupation:engineer): ")
                age, gender, occupation = parse_personal_info(user_response)
                if age is not None and gender is not None and occupation is not None:
                    logger.info(f"成功解析用户基本信息：年龄={age}, 性别={gender}, 职业={occupation}")
                    break
                else:
                    missing_fields = []
                    if age is None:
                        missing_fields.append("年龄")
                    if gender is None:
                        missing_fields.append("性别")
                    if occupation is None:
                        missing_fields.append("职业")
                    dialog_print(
                        f"无法解析您的输入，请确保包含以下信息：{', '.join(missing_fields)}，且使用逗号、顿号、空格或关键词分隔（例如：25，男，工程师 或 年龄:25, 性别:男, 职业:工程师 或 25 男 工程师）")

        identification = f"Age: {age}, Gender: {gender}, Occupation: {occupation}"
        print(f"\n用户基本信息：{identification}")

        memory_graph = MemoryGraph(identification)
        total_history.append({"role": "system", "content": identification})
        last_question = initial_message
        last_response = user_response

        for idx, topic in enumerate(topics, 1):
            dialog_print("\n")
            logger.info(f"开始处理主题 {idx}/{len(topics)}：{topic}")
            dialog_print(f"{'-'*20}当前主题: {topic}")

            memory_graph.add_topic(topic)

            asked_questions = 0
            depth = 0
            max_depth = 3
            current_topic_history = []

            while depth < max_depth:
                question_type = "initial" if depth == 0 else "followup"
                memory_context_str = memory_graph.get_context_for_prompt(topic)
                other_topics_str = ", ".join([t for t in topics if t != topic])
                examples = chatprompt.get(topic, [])
                example_questions_str = "\n".join([f"- {q}" for q in examples])
                question_payload = (
                    f"Type: {question_type}\n"
                    f"Topic: {topic}\n"
                    f"Identification: {identification}\n"
                    f"Last Question: {last_question}\n"
                    f"Last Response: {last_response}\n"
                    f"Memory: {memory_context_str}\n"
                    f"Other Topics: {other_topics_str}\n"
                    f"Examples:\n{example_questions_str}"
                )
                question = makerequest(group_chat_manager, user_proxy, question_agent, question_payload)                
                if question is None:
                    question = "抱歉，我现在无法生成问题。"

                dialog_print(f"冰糖提问: {question}")
                logger.info(f"冰糖提问: {question}")
                
                if automated:
                    current_scoring_standard = scoring_standards[scale_name][topic]
                    response = generate_mock_response(question, topic, identification, real_interview, scale_scores, scoring_standard=current_scoring_standard, 
                                                      current_topic_history=current_topic_history, depth=depth, scale_name=scale_name)
                    dialog_print(f"\n模拟回答：{response}")
                else:
                    response = get_valid_input("\n您的回答: ")
                logger.info(f"用户回答：{response}")
                scores.append({"topic": topic, "question": question, "response": response})

                group_chat_manager.groupchat.messages.append({
                    "content": response,
                    "role": "user",
                    "name": "UserProxy"
                })

                qa_count += 1

                total_history.append({"role": "assistant", "content": question})
                total_history.append({"role": "user", "content": response})
                current_topic_history.append({"question": question, "response": response})
                last_question = question
                last_response = response
                memory_graph.add_short_term_memory(topic, response, turn_id=qa_count)
                topic_history_str = "\n".join([f"Q: {qa['question']}\nA: {qa['response']}" for qa in current_topic_history])
                necessity_payload = f"Topic: {topic}\nHistory:\n{topic_history_str}"
                necessity_score_text = makerequest(group_chat_manager, user_proxy, necessity_agent, necessity_payload)

                if necessity_score_text is not None:
                    necessity_score = extract_score(necessity_score_text)
                else:
                    necessity_score = 0
                asked_questions += 1
                depth += 1
                if is_necessary(necessity_score, asked_questions):
                    pass
                else:
                    break
            topic_history_str = "\n".join([f"Q: {qa['question']}\nA: {qa['response']}" for qa in current_topic_history])
            scoring_standard_str = json.dumps(scoring_standards[scale_name][topic], ensure_ascii=False, indent=2)
            scoring_payload = f"Topic: {topic}\nHistory:\n{topic_history_str}\nStandard:\n{scoring_standard_str}"
            total_score_text = makerequest(group_chat_manager, user_proxy, scoring_agent, scoring_payload)
            if total_score_text is not None:
                total_score, summary = extract_score_and_summary(total_score_text, scale_name)
            else:
                total_score, summary = 0, ""
            dialog_print(f"\n主题'{topic}'的总评分: {total_score} 分")
            if summary:
                dialog_print(f"得分依据: {summary}\n")
            else:
                print()
            scores.append({"topic": topic, "question": "Total score", "response": "", "score": total_score})
            memory_graph.convert_topic_to_long_term(topic, total_score, summary)
        total_history_str = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in total_history
        )
        initial_scores_from_graph = {}
        for topic in topics:
            if memory_graph.graph.has_node(topic):
                node_data = memory_graph.graph.nodes[topic]
                if node_data.get('status') == 'completed':
                    initial_scores_from_graph[topic] = node_data.get('score', 0)
        scores_str_for_summary_agent = ", ".join([f"{topic}:{score}" for topic, score in initial_scores_from_graph.items()])

        final_memory_str = memory_graph.get_context_for_prompt("Overall Summary")
        summary_payload = f"Full History:\n{total_history_str}\n\nInitial Scores:\n{scores_str_for_summary_agent}\n\nMemory:\n{final_memory_str}"

        summary_output = makerequest(group_chat_manager, user_proxy, summary_agent, summary_payload)
        if summary_output is not None:
            summary, updated_scores = extract_summary_and_updated_scores(summary_output, scale_name)
        else:
            summary, updated_scores = "", {}

        if updated_scores:
            dialog_print("\n--- 分数调整 ---")
            for topic, details in updated_scores.items():
                score = details["score"]
                reason = details["reason"]
                memory_graph.update_topic_score(topic, score, reason)
                dialog_print(f"主题'{topic}'的分数已更新为: {score} 分，理由是：{reason}")
                logger.info(f"主题'{topic}'的分数已更新： -> {score} 分，理由是：{reason}")
        else:
            logger.warning("未收到更新后的分数。")
            dialog_print("\n所有主题分数均未调整。")

        overall_score = 0
        for topic in topics:
            if memory_graph.graph.has_node(topic):
                attrs = memory_graph.graph.nodes[topic]
                if attrs.get('status') == 'completed':
                    score_to_use = attrs.get('updated_score') if attrs.get('updated_score') is not None else attrs.get('score', 0)
                    if score_to_use is None:
                        logger.warning(f"主题 '{topic}' 的 score_to_use 为 None，使用默认值 0")
                        score_to_use = 0
                    overall_score += score_to_use            
        symptom_level = categorize_score(overall_score, scale_name)
        logger.info(f"总评分：{overall_score}，症状等级：{symptom_level}")
        report_table = generate_score_table(memory_graph, topics, symptom_level)
        logger.info("生成得分表格完成。")

        final_report = generate_report(report_table, summary, scale_name)
        if final_report is None:
            final_report = "抱歉，无法生成报告。"
        logger.info("评估任务完成。")
        dialog_print(f"\n本次Q&A数量：{qa_count}")
        return final_report, overall_score, symptom_level, updated_scores

    except Exception as e:
        logger.exception("执行评估任务时发生未知错误：%s", e)
        return "抱歉，评估过程中发生了错误。"




def process_single_file(file_path, scoring_standards, chatprompt, selected_scale, mode_choice, csv_file_path, automated=False):
    try:
        identifier = os.path.splitext(os.path.basename(file_path))[0] 
        if is_file_already_evaluated(identifier, csv_file_path):
            logger.info(f"文件 {file_path} 已经评估过，跳过。")
            dialog_print(f"文件 {file_path} 已经评估过，跳过。")
            return 
        logger.info(f"开始处理文件：{file_path}")
        dialog_print(f"\n{'='*50}\n开始处理文件：{file_path}\n{'='*50}\n")
        identifier, real_interview, scores = load_real_data(file_path, selected_scale)
        agents = setup_agents(chatprompt)
        if automated:
            clear_memory_response = generate_mock_response("", topic=None, identification="", real_interview=[], scale_scores={}, clear_memory=True, scoring_standard=None, current_topic_history=None)
            logger.info(f"API清除记忆的响应：{clear_memory_response}")
            dialog_print("\n已清除API的对话记忆，准备处理当前文件。\n")

        final_report, overall_score, symptom_level, updated_scores = perform_assessment(
            topics=list(chatprompt.keys()),
            chatprompt=chatprompt,
            agents=agents,
            scale_name=selected_scale,
            scoring_standards=scoring_standards,
            real_interview=real_interview,
            scale_scores=scores,
            automated=automated
        )

        save_assessment_results(
            identifier=identifier,
            overall_score=overall_score,
            symptom_level=symptom_level,
            updated_scores=updated_scores,
            csv_file=csv_file_path,
            scale_name=selected_scale
        )
        logger.info(f"评估结果已保存到 {csv_file_path}")
        dialog_print(f"评估结果已保存到 {csv_file_path}")

        if mode_choice == "2":
            dialog_print("\n自动化测试完成，生成的心理评估报告如下：\n")
        else:
            dialog_print(
                "\n感谢你的真诚配合，接下来我将为您输出一份心理初筛报告。结果仅供参考，不作为医学诊断依据。生成报告可能需要较长时间，请稍等。\n")
        dialog_print(final_report)
        logger.info("心理评估报告已输出给用户。程序结束。")

    except Exception as e:
        logger.exception(f"处理文件 {file_path} 时发生错误：{e}")
        dialog_print(f"处理文件 {file_path} 时发生错误，请查看日志获取详细信息。")

