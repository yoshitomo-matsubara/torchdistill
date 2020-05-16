import logging

PROJECT_NAME = 'kdkit'

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
def_logger = logging.getLogger(PROJECT_NAME)
