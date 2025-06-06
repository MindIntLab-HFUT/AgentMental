import json
import logging
import sys

logger = logging.getLogger(__name__)


def load_json_file(file_path: str):
    """通用 JSON 文件加载函数。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"文件 JSON 格式无效: {file_path}")
        raise


# 加载量表
def load_chatprompt(file_path="data/HAMD-17.json"):
    return load_json_file(file_path)


def load_scoring_standards(file_path="data/scoring_standards.json"):
    """加载量表评分标准。"""
    return load_json_file(file_path)


def load_real_data(file_path: str):
    """
    从JSON文件加载真实访谈对话和量表得分数据。

    :param file_path: JSON文件路径
    :return: (real_interview, hamd_17_scores) 元组
    """
    try:
        data = load_json_file(file_path)
        video_name = data.get("video_name", "unknown_video")
        real_interview = data.get("real_interview", [])
        hamd_17_scores = data.get("hamd_17_scores", {})
        return video_name,real_interview, hamd_17_scores
    except FileNotFoundError:
        logger.error(f"文件 {file_path} 未找到。")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"文件 {file_path} 不是有效的JSON格式。")
        sys.exit(1)

