class EmptyDatasetException(ValueError):
    pass


class ValidationError(ValueError):
    def __init__(self, message, errors):
        super(ValidationError, self).__init__(message)
        self.errors = errors
