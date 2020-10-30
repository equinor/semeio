import stea
import argparse
from semeio import valid_file

description = (
    "STEA is a powerful economic analysis tool used for complex economic "
    "analysis and portfolio optimization. STEA helps you analyze single "
    "projects, large and small portfolios and complex decision trees. "
    "As output, for each of the entries in the result section of the "
    "yaml config file, STEA will create result files "
    "ex: Res1_0, Res2_0, .. Res#_0"
)


category = "modeling.financial"


def _get_args_parser():

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-c",
        "--config",
        help="STEA config file, yaml format required",
        type=valid_file,
    )
    return parser


def main_entry_point():
    parser = _get_args_parser()
    options = parser.parse_args()
    stea_input = stea.SteaInput([options.config])
    res = stea.calculate(stea_input)
    for res, value in res.results(stea.SteaKeys.CORPORATE).items():
        with open("{}_0".format(res), "w") as ofh:
            ofh.write("{}\n".format(value))
