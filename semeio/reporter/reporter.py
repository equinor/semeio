class Reporter:
    def __init__(self, name):
        self.name = name

    def _get_suffix_for_value(self, value):

        if isinstance(value, (str, float, int)):
            return "txt"

        valid_types = ",".join(["string", "float", "int"])

        raise TypeError(
            "The type of value <{}> must be one of: {} was {}".format(
                value, valid_types, type(value)
            )
        )

    def report(self, key=None, value=None, write=False):
        if key is None and write:
            raise ValueError("key must be specified if write is True, was None")

        if write:
            filename = "{job_name}_{key}.{suffix}".format(
                job_name=self.name,
                key=str(key),
                suffix=self._get_suffix_for_value(value),
            )

            with open(filename, "w") as fh:
                fh.write(str(value))
        else:
            print(value)
