import numpy as np
import KernSmooth


def main():
    x = np.array([1, 2, 3, 4, 5])
    y = KernSmooth.bkde(x)
    print(y)


if __name__ == "__main__":
    main()
