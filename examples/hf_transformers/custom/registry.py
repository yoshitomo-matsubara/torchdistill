CONFIG_DICT = dict()
TOKENIZER_DICT = dict()


def register_config_class(cls):
    CONFIG_DICT[cls.__name__] = cls
    return cls


def register_tokenizer_class(cls):
    TOKENIZER_DICT[cls.__name__] = cls
    return cls
