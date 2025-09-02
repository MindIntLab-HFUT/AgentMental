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
base_url = os.getenv("API_BASE_URL", "http://localhost:10010/v1/")
api_key = os.getenv("API_KEY", "none")
model_name = os.getenv("API_MODEL", "deepseek-r1-32b")


def generate_mock_response(question, topic, identification, real_interview, scale_scores, scoring_standard=None, current_topic_history=None, depth=0, clear_memory=False, scale_name="PHQ-8"):
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
Please provide your basic information (age, gender, occupation) in the format 'age:<age>, gender:<gender>, occupation:<occupation>' (e.g., age:25, gender:male, occupation:engineer). If the dialogue does not explicitly mention age, gender, or occupation, infer plausible values based on the context of the dialogue (e.g., tone, content, or implied demographics). For gender, only use one of the following options: 'male', 'female', or 'other'. For age, use a realistic value between 0 and 80.
Ensure the response is concise and follows the requested format without adding anything else.\n
"""
        logger.info(f"Prompt for generating simulated reply: {prompt}")
    else:
        if depth == 0:
            prompt = f"""Please answer the following question:\n{question}\nPlease provide a truthful and reasonable answer based on your real interview dialogue and your profile, with no more than 50 words:"""
        else:
            scoring_standard_str = "\n".join([f"{k} - {v}" for k, v in scoring_standard.items()])
            topic_history_str = ""
            if current_topic_history:
                for qa in current_topic_history:
                    topic_history_str += f"Question: {qa['question']}\nUser's Response: {qa['response']}\n"

            prompt = f"""Please answer the following in-depth question:\n{question}\n
Please provide a truthful and reasonable answer based on the interview dialogue and all your previous responses. Do not fabricate symptom situations in your response. Your previous responses: \n{topic_history_str}\n
When the question content exceeds the scope of the interview dialogue and cannot be accurately answered, you may choose to answer 'not sure' or a similar expression, but use it sparingly and provide a reason that fits the patient's information.
Limit your response to 50 words.
"""
        logger.info(f"Prompt for generating simulated reply: {prompt}")

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        system_prompt = f"""You are speaking with a psychological assistant, and you are the client in this conversation with the following interview dialogue:
{interview_history}

You should follow the provided information to act as a client in the conversation. Your responses should be coherent and avoid repeating previous utterances.
Your response should ONLY include what the Client should say, in a natural, first-person tone.
""" 
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        response = completion.choices[0].message.content
        logger.info(f"Generated simulated reply: {response}")

        think_index = response.find("</think>")
        if think_index != -1:
            response = response[think_index + len("</think>") :].strip()

        return response

    except Exception as e:
        ogger.exception(f"Error while calling API to generate reply: {e}")
        return "Sorry, I cannot answer this question at the moment."