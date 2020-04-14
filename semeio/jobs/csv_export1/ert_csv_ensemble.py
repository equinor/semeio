import os

from cwrap import open as copen
from semeio.jobs.csv_export1.ert_ensemble import ErtEnsemble
from semeio.jobs.csv_export1.export import Export


class ErtCSVEnsemble(ErtEnsemble):
    def load_data(self, mem):
        export = Export(mem.iens, mem.iteration, mem.path, mem.case, lazy_load=False)
        mem.data = export

    def loadParameters(self, filename=None):
        for member in self:
            member.data.loadParameters(filename)

    def loadDesignMatrix(self, design_matrix):
        for mem in self:
            mem.data.loadDesignMatrix(design_matrix)

    def addSummaryPattern(self, key_pattern):
        for mem in self:
            mem.data.addSummaryPattern(key_pattern)

    def dump(self, ens_export_file, dateInterval=None):
        (path, name) = os.path.split(ens_export_file)
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
        ensemble_fileH = copen(ens_export_file, "w")

        header_written = False
        for mem in self:
            if not header_written:
                header_written = mem.data.dumpHeader(ensemble_fileH)

            mem.data.dump(ensemble_fileH, dateInterval)

        ensemble_fileH.close()
