import autogen
import os

def get_llm_config():
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
        file_location=".",
        filter_dict={"model": ["qwen2.5-72b"]}     # qwen2.5:72b-instruct
    )

    llm_config = {
        "config_list": config_list,
        "cache_seed": 42,
        "temperature": 0,
        "max_tokens": 2048
    }
    
    return llm_config