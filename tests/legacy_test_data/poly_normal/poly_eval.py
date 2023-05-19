#!/usr/bin/env python
import json
from pathlib import Path

# pylint: disable=invalid-name


def _load_coeffs(filename):
    return json.load(Path(filename).read_text(encoding="utf-8"))


def _evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


if __name__ == "__main__":
    coefficients = _load_coeffs("coeffs.json")
    output = [_evaluate(coefficients, x) for x in range(10)]
    Path("poly_0.out").write_text("\n".join(map(str, output)), encoding="utf-8")
