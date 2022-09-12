#!/usr/bin/env python
# pylint: disable=consider-using-f-string
import os
import sys

iens = None
if len(sys.argv) > 1:
    iens = int(sys.argv[1])

numbers = [1, 2, 3, 4, 5, 6]
if iens in numbers:
    random_report_step = (numbers.index(iens) % 3) + 1
    os.remove("perlin_%d.txt" % random_report_step)

    with open("perlin_fail.status", "w", encoding="utf-8") as f:
        f.write("Deleted report step: %d" % random_report_step)

else:
    with open("perlin_fail.status", "w", encoding="utf-8") as f:
        f.write("Did nothing!")
