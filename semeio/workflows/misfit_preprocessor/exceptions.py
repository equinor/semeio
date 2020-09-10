class ValidationError(ValueError):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
        self.message = message

    def __str__(self):
        msg = "{}\n".format(self.message)
        for error in self.errors:
            path = ".".join(error.key_path) if len(error.key_path) > 0 else "root level"
            msg += "  - {} ({})\n".format(error.msg, path)
        return msg
