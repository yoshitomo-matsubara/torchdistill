class ForwardTerminationException(Exception):
    def __init__(self, *args):
        pass

    def __str__(self):
        return 'ForwardTerminationException has been raised'
