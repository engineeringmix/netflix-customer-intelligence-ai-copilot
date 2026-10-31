"""Guard against invented numbers: every figure in LLM prose must appear in the evidence that SQL actually returned."""
import re

import numpy as np
import pandas as pd

_NUM = re.compile(r"(?<![\w.])\$?-?\d[\d,]*(?:\.\d+)?%?")


def _walk(x, out):
    if isinstance(x, dict):
        for v in x.values():
            _walk(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _walk(v, out)
    elif isinstance(x, pd.DataFrame):
        _walk(x.values.tolist(), out)
    elif isinstance(x, (int, float, np.integer, np.floating)) and not (isinstance(x, float) and np.isnan(x)):
        v = float(x)
        for scale in (1, 100):                       # a fraction 0.256 may be quoted as 25.6% or 26%
            for d in (0, 1, 2):
                out.add(round(abs(v * scale), d))
        out.add(abs(v))
    elif isinstance(x, str):
        for m in _NUM.findall(x):
            try:
                out.add(abs(float(m.replace(",", "").replace("$", "").replace("%", ""))))
            except ValueError:
                pass


def allowed_numbers(evidence) -> set:
    out = set()
    _walk(evidence, out)
    return out


def unverified_numbers(text: str, evidence, tolerate_small: bool = True) -> list:
    """Numbers in `text` that are not present (to the shown precision) in `evidence`."""
    allowed = allowed_numbers(evidence)
    bad = []
    for m in _NUM.findall(text):
        raw = m.replace(",", "").replace("$", "").replace("%", "")
        try:
            v = abs(float(raw))
        except ValueError:
            continue
        if tolerate_small and v <= 10 and float(raw).is_integer():     # "3 themes", "2 months"
            continue
        if 1990 <= v <= 2100 and float(raw).is_integer():              # years
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        if not any(abs(round(a, dec) - v) < 1e-9 or abs(a - v) < 0.5 * 10 ** (-dec) + 1e-9 for a in allowed):
            bad.append(m)
    return bad
