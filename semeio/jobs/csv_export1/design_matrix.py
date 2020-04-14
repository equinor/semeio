from collections import OrderedDict


class DesignMatrix(object):
    def __init__(self):
        super(DesignMatrix, self).__init__()
        self.__data = {}

    def addRealizationNumber(self, realization_number):
        if not realization_number in self.__data:
            self.__data[realization_number] = OrderedDict()

    def __getitem__(self, realization_number):
        if not realization_number in self.__data:
            raise IndexError("Realization number %d is not added." % realization_number)
        return self.__data[realization_number]

    def __iter__(self):
        keys = self.__data.keys()
        keys = sorted(keys)

        for key in keys:
            yield self[key]

    def __contains__(self, item):
        return item in self.__data
    def dumpLine(self, fileH, iens):
        data_line = self[iens]
        for key in data_line.keys():
            fileH.write(",%s" % data_line[key])
