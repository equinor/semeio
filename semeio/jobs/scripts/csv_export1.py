#!/usr/bin/env python

import sys
from semeio.jobs.csv_export1 import csv_export1

if __name__ == "__main__":
    csv_export1.export(*sys.argv[1:])
