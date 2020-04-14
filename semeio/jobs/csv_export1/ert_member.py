import sys
import glob
import os.path


class ErtMember:
    def __init__(self, iens, path, case, iteration):
        self.path = path
        self.iens = iens
        self.case = case
        self.iteration = iteration
        self.data = None
        self.active = True

    def grid_file(self):
        return "%s/%s.EGRID" % (self.path, self.case)

    def init_file(self):
        return "%s/%s.INIT" % (self.path, self.case)

    def removeMatching(self, glob_pattern):
        for file in glob.glob(os.path.join(self.path, glob_pattern)):
            try:
                os.unlink(file)
            except OSError:
                sys.stderr.write("** ERROR: Failed to remove file:%s" % file)

    def __equal__(self, other):
        if self.iens == other.iens and self.iteration == other.iteration:
            return True
        else:
            return False

    def __nonzero__(self):
        if isinstance(self.data, bool):
            return self.data
        else:
            return self.data is not None

    def __str__(self):
        return "path:%s  eclbase:%s  realization:%d  iteration:%d" % (
            self.path,
            self.case,
            self.iens,
            self.iteration,
        )

    def getData(self):
        return self.data
