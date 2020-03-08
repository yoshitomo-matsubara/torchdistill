class ForwardTerminationException(Exception):
    def __init__(self, *args):
        self.message = args[0] if args is not None and len(args) > 0 else None

    def __str__(self):
        return 'ForwardTerminationException{}'.format(': {}'.format(self.message) if self.message is not None
                                                      else ' has been raised')
