#!/usr/bin/env python
from resdata.summary import Summary


def write_diff(filename, summary, key1, key2):
    with open(filename, "w", encoding="utf-8") as file:
        for vec_1, vec_2 in zip(summary.numpy_vector(key1), summary.numpy_vector(key2)):
            diff = vec_1 - vec_2
            file.write(f"{diff:f}\n")


if __name__ == "__main__":
    SUM = Summary("SNAKE_OIL_FIELD")
    REPORT_STEP = 199

    write_diff(f"snake_oil_opr_diff_{REPORT_STEP}.txt", SUM, "WOPR:OP1", "WOPR:OP2")
    write_diff(f"snake_oil_wpr_diff_{REPORT_STEP}.txt", SUM, "WWPR:OP1", "WWPR:OP2")
    write_diff(f"snake_oil_gpr_diff_{REPORT_STEP}.txt", SUM, "WGPR:OP1", "WGPR:OP2")
