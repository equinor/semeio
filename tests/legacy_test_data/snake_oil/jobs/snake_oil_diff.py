#!/usr/bin/env python
from ecl.summary import EclSum


def write_diff(filename, ecl_sum, key1, key2):
    with open(filename, "w", encoding="utf-8") as file:
        for vec_1, vec_2 in zip(ecl_sum.numpy_vector(key1), ecl_sum.numpy_vector(key2)):
            diff = vec_1 - vec_2
            file.write(f"{diff:f}\n")


if __name__ == "__main__":
    ECL_SUM = EclSum("SNAKE_OIL_FIELD")
    REPORT_STEP = 199

    write_diff(f"snake_oil_opr_diff_{REPORT_STEP}.txt", ECL_SUM, "WOPR:OP1", "WOPR:OP2")
    write_diff(f"snake_oil_wpr_diff_{REPORT_STEP}.txt", ECL_SUM, "WWPR:OP1", "WWPR:OP2")
    write_diff(f"snake_oil_gpr_diff_{REPORT_STEP}.txt", ECL_SUM, "WGPR:OP1", "WGPR:OP2")
