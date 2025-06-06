import os
import sys
import logging
import pandas as pd
from config import get_llm_config
from data_load import load_chatprompt, load_scoring_standards, load_real_data
from logging_setup import setup_logging, initialize_dialog_log, dialog_print, close_dialog_log
from utils import get_valid_input, choose_mode
from assessment import process_single_file




if __name__ == "__main__":
    try:
        logger = setup_logging()
        initialize_dialog_log()
        logger.info("心理评估程序启动。")

        data_dir = "data/depression_data_111"
        if not os.path.exists(data_dir):
            logger.error(f"数据文件夹 {data_dir} 不存在。")
            dialog_print(f"错误：数据文件夹 {data_dir} 不存在。")
            sys.exit(1)

        # 获取所有JSON文件
        json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
        if not json_files:
            logger.error(f"在文件夹 {data_dir} 中未找到任何JSON文件。")
            dialog_print(f"错误：在文件夹 {data_dir} 中未找到任何JSON文件。")
            sys.exit(1)

        logger.info(f"在文件夹 {data_dir} 中找到 {len(json_files)} 个JSON文件。")
        dialog_print(f"在文件夹 {data_dir} 中找到 {len(json_files)} 个JSON文件。")


        # 加载评分标准
        scoring_standards = load_scoring_standards("scoring_standards.json")

        # 选择量表
        available_scales = list(scoring_standards.keys())
        dialog_print("请选择要使用的心理评估量表：")
        for idx, scale in enumerate(available_scales, 1):
            print(f"{idx}. {scale}")

        while True:
            scale_choice = get_valid_input(f"请输入选择的量表编号（1-{len(available_scales)}）： ")
            if scale_choice.isdigit():
                scale_idx = int(scale_choice)
                if 1 <= scale_idx <= len(available_scales):
                    selected_scale = available_scales[scale_idx - 1]
                    break
            dialog_print("无效的选择，请重新输入。")

        logger.info(f"选择的量表为：{selected_scale}")
        dialog_print(f"您选择的量表是：{selected_scale}")

        # 加载 chatprompt
        if selected_scale == "HAMA":
            chatprompt = load_chatprompt("HAMA.json")
        elif selected_scale == "HAMD-17":
            chatprompt = load_chatprompt("HAMD-17.json")
        else:
            dialog_print("未支持的量表类型。")
            sys.exit(1)

        # 选择是否启用自动化测试
        mode_choice = choose_mode()

        if mode_choice == "2":
            print("启用自动化测试模式。")
            automated = True
        else:
            automated = False

        # 指定CSV文件路径
        csv_file_path = "evaluation/72b.csv"


        # 遍历并处理每个JSON文件
        for json_file in json_files:
            process_single_file(json_file, scoring_standards, chatprompt, selected_scale, mode_choice, csv_file_path, automated)

    except Exception as e:
        logger.exception("程序运行过程中发生未捕获的错误：%s", e)
        dialog_print("程序运行过程中发生错误，请检查日志获取详细信息。")

    finally:
        close_dialog_log()
