import logging
import builtins
import sys

def setup_logger():
    logger = logging.getLogger("my_logger")  # 使用自定义的 logger 名称
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    
    # 创建格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # 创建控制台处理器 (StreamHandler) 并设置格式
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 添加处理器到 logger
    logger.addHandler(console_handler)

    builtin_print = builtins.print
    def print(*args, **kwargs):
        kwargs['flush'] = True
        # 检查是否在处理异常，如果是则使用原始 print 函数
        if sys.exc_info()[0] is None:
            logger.info(" ".join(str(arg) for arg in args), stacklevel=2)
        else:
            builtin_print(*args, **kwargs)
    builtins.print = print
    return logger

# 初始化 logger
logger = setup_logger()