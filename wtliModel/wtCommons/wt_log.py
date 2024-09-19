import logging
import datetime


def wt_log():
    # 创建日志器
    logger = logging.getLogger('multi_level_logger')
    logger.setLevel(logging.DEBUG)  # 设置最低日志级别
    log_time = datetime.datetime.now().strftime("%Y%m%d")
    # 创建文件处理器
    debug_handler = logging.FileHandler(
        rf'E:\workspace_python311\project-thymoma\wtlizzz-core\wtliModel\wtliNewModel\log\debug-{log_time}.log')
    info_handler = logging.FileHandler(
        rf'E:\workspace_python311\project-thymoma\wtlizzz-core\wtliModel\wtliNewModel\log\info-{log_time}.log')
    error_handler = logging.FileHandler(
        rf'E:\workspace_python311\project-thymoma\wtlizzz-core\wtliModel\wtliNewModel\log\error-{log_time}.log')

    # 设置处理器的日志级别
    debug_handler.setLevel(logging.DEBUG)
    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.ERROR)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # 将处理器添加到日志器
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger


wt_log = wt_log()


class Wt_logger_manager:
    def __init__(self, name):
        # 创建日志器
        self.logger = logging.getLogger('multi_level_logger')
        self.logger.setLevel(logging.DEBUG)  # 设置最低日志级别
        self.log_time = datetime.datetime.now().strftime("%Y%m%d")
        # 创建文件处理器
        debug_handler = logging.FileHandler(
            rf'E:\workspace_python311\project-thymoma\wtlizzz-core\wtliModel\wtliNewModel\log\{name}-debug-{self.log_time}.log')
        info_handler = logging.FileHandler(
            rf'E:\workspace_python311\project-thymoma\wtlizzz-core\wtliModel\wtliNewModel\log\{name}-info-{self.log_time}.log')
        error_handler = logging.FileHandler(
            rf'E:\workspace_python311\project-thymoma\wtlizzz-core\wtliModel\wtliNewModel\log\{name}-error-{self.log_time}.log')

        # 设置处理器的日志级别
        debug_handler.setLevel(logging.DEBUG)
        info_handler.setLevel(logging.INFO)
        error_handler.setLevel(logging.ERROR)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        debug_handler.setFormatter(formatter)
        info_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)

        # 将处理器添加到日志器
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(error_handler)

    def get_logger(self):
        return self.logger
