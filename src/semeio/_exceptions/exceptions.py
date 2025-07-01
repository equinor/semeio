class ConfigurationError(Exception):
    def __init__(self, message: str, errors: Exception):
        super().__init__(message)
        self.errors = errors
        self.message = message

    def __str__(self) -> str:
        msg = f"{self.message}"
        for error in self.errors:
            msg += f"\n{error}"
        return msg


class ValidationError(Exception):
    pass
