class ConfigurationError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
        self.message = message

    def __str__(self):
        msg = "{}".format(self.message)
        for error in self.errors:
            msg += "\n{}".format(error)
        return msg


class ValidationError(Exception):
    pass
