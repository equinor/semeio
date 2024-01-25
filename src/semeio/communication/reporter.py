import json
import os


class FileReporter:
    def __init__(self, output_dir):
        self._output_dir = output_dir

        if not os.path.isabs(output_dir):
            raise ValueError(
                f"Expected output_dir to be an absolute path, received '{output_dir}'"
            )

    def publish_csv(self, namespace, data):
        output_file = self._prepare_output_file(namespace) + ".csv"
        data.to_csv(output_file)

    def publish(self, namespace, data):
        output_file = self._prepare_output_file(namespace) + ".json"

        if os.path.exists(output_file):
            with open(output_file, encoding="utf-8") as f_handle:
                all_data = json.load(f_handle)
        else:
            all_data = []

        all_data.append(data)
        with open(output_file, "w", encoding="utf-8") as f_handle:
            json.dump(all_data, f_handle, default=str)

    def publish_msg(self, namespace, msg):
        output_file = self._prepare_output_file(namespace)
        with open(output_file, "a", encoding="utf-8") as f_handle:
            f_handle.write(f"{msg}\n")

    def _prepare_output_file(self, namespace):
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        if not os.path.isdir(self._output_dir):
            raise ValueError(
                f"Expected output_dir to be a directory ({self._output_dir})"
            )

        if os.path.sep in namespace:
            raise ValueError(f"Namespace contains path separators ({namespace})")

        return os.path.join(self._output_dir, namespace)
