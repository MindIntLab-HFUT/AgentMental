import os
import sys
import logging
from datetime import datetime

dialog_log_file = None

def setup_logging(log_dir='logs'):
    """
    设置日志记录器，日志将存储在指定的子文件夹中。
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, log_dir)


    if not os.path.exists(log_path):
        os.makedirs(log_path)


    log_filename = datetime.now().strftime("log_%Y%m%d_%H%M%S.log")
    log_file = os.path.join(log_path, log_filename)

    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别DEBUG < INFO < WARNING < ERROR < CRITICAL
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

    logger = logging.getLogger()
    logger.info("日志系统已初始化。")
    return logger

def initialize_dialog_log(log_dir='dialog_logs'):
    """
    初始化对话日志文件，日志将存储在指定的子文件夹中。
    """
    global dialog_log_file  

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, log_dir)


    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_filename = datetime.now().strftime("dialog_72b_%Y%m%d_%H%M%S.txt")
    log_file = os.path.join(log_path, log_filename)

    dialog_log_file = open(log_file, 'w', encoding='utf-8')
    print(f"对话日志系统已初始化，日志文件：{log_file}")

def dialog_print(text):
    """
    将文本输出到控制台和对话日志文件。
    """
    print(text)  
    if dialog_log_file:
        dialog_log_file.write(text + '\n')  
        dialog_log_file.flush()  

def close_dialog_log():
    """关闭对话日志文件。"""
    global dialog_log_file
    if dialog_log_file:
        dialog_log_file.close()
        dialog_log_file = None
        print("对话日志文件已关闭。")