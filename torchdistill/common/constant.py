import logging

LOGGING_FORMAT = '%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s'

logging.basicConfig(
    format=LOGGING_FORMAT,
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
def_logger = logging.getLogger()
