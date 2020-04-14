import sys
from datetime import datetime, time
import os.path

from ecl.summary import EclSumKeyWordVector
from ecl.summary import EclSum
from semeio.jobs.csv_export1.design_matrix import DesignMatrix


class SumCase(object):
    sum_header_fmt = " %17s "
    sum_fmt = " %17.5f "

    def __init__(self, case, slave, lazy_load=True):
        self.__sum = EclSum(case, lazy_load=lazy_load)
        self.__keywords = None

        self.__slave = slave
        if slave:
            self.__key_prefix = self.__sum.base
        else:
            self.__key_prefix = ""

    def __len__(self):
        if self.__keywords:
            return len(self.__keywords)
        else:
            return 0

    def getEclSum(self):
        return self.__sum

    def copyKeys(self, other):
        other.__keywords = self.__keywords.copy(other.getEclSum())

    def initKeys(self, pattern_list):
        if not self.__keywords:
            keys = []
            for key_pattern in pattern_list:
                for key in self.__sum.keys(key_pattern):
                    keys.append(key)
            keys.sort()

            self.__keywords = EclSumKeyWordVector(self.__sum)
            for key in keys:
                self.__keywords.addKeyword(key)

    def dumpHeader(self, outputH):
        for key in self.__keywords:
            key = key.replace(",", "_")

            if self.__slave:
                outputH.write(",%s:%s" % (self.__key_prefix, key))
            else:
                outputH.write(",%s" % key)

    def dumpLine(self, outputH, time_value):
        self.__sum.dumpCSVLine(time_value, self.__keywords, outputH)


class Export(object):
    date_fmt = "%Y-%m-%d"  # To be used with strftime

    def __init__(self, iens, iteration, path, input_case, sum_keys=[], lazy_load=True):
        case = os.path.join(path, input_case)
        self.__iens = int(iens)
        self.__iteration = int(iteration)
        self.__params = {}
        self.__param_names = []
        self.__case_list = []
        self.__lazy_load = lazy_load

        self.addCase(case, False)
        self.__masterCase = self.__case_list[0].getEclSum()
        self.__path = path

        self.__sum_patterns = []
        for key in sum_keys:
            self.addSummaryPattern(key)

    def addCase(self, case, slave):
        self.__case_list.append(SumCase(case, slave, lazy_load=self.__lazy_load))

    def addSlave(self, case):
        fullPath = os.path.join(self.__path, case)
        self.addCase(fullPath, True)

    def addSummaryPattern(self, key_pattern):
        self.__sum_patterns.append(key_pattern)

    def __getitem__(self, key):
        return self.__params[key]

    def getPath(self):
        return self.__path

    def loadParameters(self, param_file=None):
        if param_file is None:
            param_file = "parameters.txt"

        full_path = os.path.join(self.__path, param_file)

        if os.path.isfile(full_path):
            with open(full_path) as fileH:
                for line in fileH.readlines():
                    strip_line = line.strip()
                    if strip_line:
                        tmp = line.split()
                        key = tmp[0]
                        try:
                            value = float(tmp[1])
                        except ValueError:
                            value = tmp[1]

                        self.__params[key] = value
                        self.__param_names.append(key)

    def loadDesignMatrix(self, design_matrix):
        """ @type design_matrix: DesignMatrix """
        if self.__iens in design_matrix:
            row = design_matrix[self.__iens]
            for key in row.keys():
                self.__params[key] = row[key]
                self.__param_names.append(key)

    def initKeys(self):
        master_case = self.__case_list[0]
        master_case.initKeys(self.__sum_patterns)

        for case in self.__case_list[1:]:
            if case.getEclSum():
                master_case.copyKeys(case)
            else:
                return False
        self.__param_names.sort()
        return True

    def dumpHeader(self, outputH):
        case_has_keys = self.initKeys()
        if case_has_keys:
            outputH.write("Realization,Iteration,Date")

            for param in self.__param_names:
                outputH.write(",%s" % param)

            for case in self.__case_list:
                case.dumpHeader(outputH)

            outputH.write("\n")
        return case_has_keys

    def dump(self, output, date_interval=None):
        if isinstance(output, str):
            if os.path.isabs(output):
                outputH = open(output, "w")
            else:
                outputH = open(os.path.join(self.__path, output), "w")
        else:
            outputH = output

        self.initKeys()
        time_list = self.findDates(date_interval)
        for time in time_list:
            self.dumpLine(outputH, time)

    def findDates(self, date_interval):
        case = self.__masterCase

        if date_interval is None:
            return case.dates
        else:
            dates = case.timeRange(interval=date_interval, extend_end=False)
            datelist = []
            for d in dates:
                datelist.append(d.datetime())
            return datelist

    def dumpLine(self, outputH, date):
        outputH.write(
            "%s,%s,%s" % (self.__iens, self.__iteration, date.strftime(self.date_fmt))
        )

        for param in self.__param_names:
            value = self.__params[param]
            outputH.write(",%s" % value)

        for case in self.__case_list:
            if len(case):
                outputH.write(",")
                case.dumpLine(outputH, date)

        outputH.write("\n")
