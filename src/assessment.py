import pandas as pd
import logging
import os
from utils import get_valid_input, categorize_score, extract_score_and_summary, extract_summary_and_updated_scores, generate_score_table, is_necessary, makerequest, generate_mock_response, parse_personal_info, extract_score, save_assessment_results, is_file_already_evaluated
from prompts import build_question_prompt, build_necessity_prompt, build_scoring_prompt, build_summary_prompt, build_report_prompt
from agents import setup_agents
from data_load import load_real_data
from logging_setup import dialog_print


logger = logging.getLogger(__name__)


def perform_assessment(topics, chatprompt, agents, scale_name, scoring_standards, real_interview, hamd_17_scores, automated=False):
    try:
        logger.info("开始执行心理评估任务。")
        question_agent, scoring_agent, necessity_agent, summary_agent, report_agent = agents

        total_history = []
        scores = []
        identification = ""
        last_topic = "基本信息收集"
        last_question = ""
        last_response = ""
        memory = []
        qa_count = 0

        initial_message = (
            f"您好, 我是您的专属心理助手冰糖。我将依据{scale_name}与您进行访谈，来评估相关症状的严重程度。请注意, 这只是一个初步的筛查, 不能代替正规的精神科诊断和治疗。 "
            "首先为了评估的准确性，我想收集您的基本信息：年龄、性别、职业，如果可以，我们开始吧。"
        )
        dialog_print(f"冰糖提问: {initial_message}")
        logger.info(f"发送初始消息：{initial_message}")
        logger.info("初始消息已发送给用户。")

        if automated:
            user_response = generate_mock_response(initial_message, topic=None, identification="", real_interview=real_interview, hamd_17_scores=hamd_17_scores,
                                                    scoring_standard=None, current_topic_history=None)
            dialog_print(f"模拟回答：{user_response}")
            age, gender, occupation = parse_personal_info(user_response)
            if age is None or gender is None or occupation is None:
                logger.error("自动化模式下，无法正确解析基本信息。")
                return "自动化模式下，无法正确解析基本信息。"
        else:
            while True:
                user_response = get_valid_input("您的回答（例如：25，男，工程师 或 年龄:25, 性别:男, 职业:工程师）： ")
                dialog_print(f"您的回答（例如：25，男，工程师 或 年龄:25, 性别:男, 职业:工程师）： {user_response}")
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

        identification = f"年龄：{age}，性别：{gender}，职业：{occupation}"
        total_history.append({"role": "system", "content": identification})
        last_question = initial_message
        last_response = user_response

        for idx, topic in enumerate(topics, 1):
            dialog_print("\n")
            logger.info(f"开始处理主题 {idx}/{len(topics)}：{topic}")
            dialog_print(f"{'-'*20}当前主题: {topic}")

            asked_questions = 0
            depth = 0
            max_depth = 3

            current_topic_history = []

            while depth < max_depth:
                prompt = build_question_prompt(topic, last_question, last_response, identification, chatprompt, memory, topics, depth=depth)
                question = makerequest(question_agent, prompt)
                if question is None:
                    question = "抱歉，我现在无法生成问题。"

                dialog_print(f"冰糖提问: {question}")
                logger.info(f"冰糖提问: {question}")

                if automated:
                    current_scoring_standard = scoring_standards[scale_name][topic]
                    response = generate_mock_response(question, topic, identification, real_interview, hamd_17_scores, scoring_standard=current_scoring_standard, current_topic_history=current_topic_history, depth=depth)
                    dialog_print(f"模拟回答：{response}")
                else:
                    response = get_valid_input("您的回答: ")
                    dialog_print(f"您的回答：{response}")
                logger.info(f"用户回答：{response}")
                scores.append({"topic": topic, "question": question, "response": response})

                total_history.append({"role": "assistant", "content": question})
                total_history.append({"role": "user", "content": response})
                current_topic_history.append({"question": question, "response": response})

                # 更新 last_question 和 last_response
                last_question = question
                last_response = response

              
                necessity_prompt = build_necessity_prompt(topic, current_topic_history)

                necessity_score_text = makerequest(necessity_agent, necessity_prompt)
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
                
                qa_count += 1


            current_scoring_standard = scoring_standards[scale_name]
            score_prompt = build_scoring_prompt(topic, current_topic_history, scale_name, current_scoring_standard)
            total_score_text = makerequest(scoring_agent, score_prompt)

            if total_score_text is not None:
                total_score, summary = extract_score_and_summary(total_score_text)
            else:
                total_score, summary = 0, ""

            dialog_print(f"\n主题'{topic}'的总评分: {total_score} 分")
            if summary:
                dialog_print(f"得分依据: {summary}\n")
            else:
                print()

            scores.append({"topic": topic, "question": "总评分", "response": "", "score": total_score})

            memory.append({
                "topic": topic,
                "score": total_score,
                "summary": summary
            })

        total_history_str = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in total_history
        )
        scores_str = ", ".join([f"{item['topic']}:{item['score']}" for item in scores if 'score' in item])

        summary_prompt = build_summary_prompt(total_history_str, scores_str, memory)
        summary_output = makerequest(summary_agent, summary_prompt)
        if summary_output is not None:
            summary, updated_scores = extract_summary_and_updated_scores(summary_output)
        else:
            summary, updated_scores = "", {}

        if updated_scores:
            for entry in memory:
                topic = entry["topic"]
                if topic in updated_scores:
                    entry["updated_score"] = updated_scores[topic]["score"]
                    entry["reason"] = updated_scores[topic]["reason"] # 同时保存更新理由
                    dialog_print(f"主题'{topic}'的分数已更新为: {updated_scores[topic]['score']} 分，理由是：{updated_scores[topic]['reason']}")
                    logger.info(f"主题'{topic}'的分数已更新： -> {updated_scores[topic]['score']} 分，理由是：{updated_scores[topic]['reason']}")
                else:
                    dialog_print(f"主题'{topic}'没有对应的更新分数。")
        else:
            logger.warning("未收到更新后的分数。")

        overall_score = sum(item.get("updated_score", item["score"]) for item in memory)
        symptom_level = categorize_score(overall_score, scale_name)
        logger.info(f"总评分：{overall_score}，症状等级：{symptom_level}")

        report_table = generate_score_table(memory, symptom_level)
        logger.info("生成得分表格完成。")

        report_prompt = build_report_prompt(report_table, summary, symptom_level)
        final_report = makerequest(report_agent, report_prompt)
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
        video_name = os.path.splitext(os.path.basename(file_path))[0] 

        if is_file_already_evaluated(video_name, csv_file_path):
            logger.info(f"文件 {file_path} 已经评估过，跳过。")
            dialog_print(f"文件 {file_path} 已经评估过，跳过。")
            return 


        logger.info(f"开始处理文件：{file_path}")
        dialog_print(f"\n{'='*50}\n开始处理文件：{file_path}\n{'='*50}\n")

        video_name, real_interview, hamd_17_scores = load_real_data(file_path)

        agents = setup_agents(chatprompt)

        if automated:
            clear_memory_response = generate_mock_response("", topic=None, identification="", real_interview=[], hamd_17_scores={}, clear_memory=True, scoring_standard=None, current_topic_history=None)
            logger.info(f"API清除记忆的响应：{clear_memory_response}")
            dialog_print("\n已清除API的对话记忆，准备处理当前文件。\n")

        final_report, overall_score, symptom_level, updated_scores = perform_assessment(
            topics=list(chatprompt.keys()),
            chatprompt=chatprompt,
            agents=agents,
            scale_name=selected_scale,
            scoring_standards=scoring_standards,
            real_interview=real_interview,
            hamd_17_scores=hamd_17_scores,
            automated=automated
        )

        save_assessment_results(
            video_name=video_name,
            overall_score=overall_score,
            symptom_level=symptom_level,
            updated_scores=updated_scores,
            csv_file=csv_file_path
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

