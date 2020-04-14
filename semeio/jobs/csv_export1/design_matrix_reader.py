from semeio.jobs.csv_export1.design_matrix import DesignMatrix


class DesignMatrixReader(object):
    @staticmethod
    def loadDesignMatrix(filename):
        """@rtype: DesignMatrix"""

        with open(filename, "r") as f:
            header = f.readline()
            keys = header.split()[1:]

            lines = f.readlines()
            design_matrix = DesignMatrix()
            for line in lines[0:]:
                line_tokens = list(map(DesignMatrixReader.convertValue, line.split()))
                if line_tokens:
                    iens = line_tokens[0]
                    design_matrix.addRealizationNumber(iens)

                    for index, value in enumerate(line_tokens[1:]):
                        key = keys[index]
                        design_matrix[iens][key] = value

        return design_matrix

    @staticmethod
    def convertValue(value):
        try:
            value = float(value)
        except ValueError:
            pass
            # value = value.rstrip()

        return value
