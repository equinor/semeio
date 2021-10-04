class EmptyDatasetException(ValueError):
    pass


class ValidationError(ValueError):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
        self.message = message

    def __str__(self):
        msg = f"{self.message}\n"
        for error in self.errors:
            msg += f"\t{error}"
        return msg
