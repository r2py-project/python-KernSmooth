import numpy as np
import r2py_kernsmooth


def main():
    x = np.array([1, 2, 3, 4, 5])
    y = r2py_kernsmooth.bkde(x)
    print(y)


if __name__ == "__main__":
    main()
