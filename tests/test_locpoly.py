## from Peter Dalgaard 2020-11-24
## '(without the truncate=FALSE, curves pretty much go through
##   the penultimate point, whatever reasonable bandwith is chosen.)'

import csv
import io
import urllib.request

import numpy as np
import r2py_kernsmooth


def load_prestige():
    """Load the Prestige dataset from the carData R package via Rdatasets."""
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Prestige.csv"
    response = urllib.request.urlopen(url, timeout=10)
    content = response.read().decode()
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    income = np.array([float(row["income"]) for row in rows])
    prestige = np.array([float(row["prestige"]) for row in rows])
    return income, prestige


def main():
    income, prestige = load_prestige()

    result_truncate = r2py_kernsmooth.locpoly(income, prestige, bandwidth=5000)
    print(result_truncate)

    result_no_truncate = r2py_kernsmooth.locpoly(
        income, prestige, bandwidth=5000, truncate=False
    )
    print(result_no_truncate)


if __name__ == "__main__":
    main()
