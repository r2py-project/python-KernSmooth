# Regression test: failed in bkfe with exact powers of 2 prior to 2.23-5
import numpy as np
import KernSmooth


def main():
    x = np.arange(1, 101, dtype=np.float64)
    print(KernSmooth.dpik(x, gridsize=256))

    x = np.array(
        [0.036, 0.042, 0.052, 0.216, 0.368, 0.511, 0.705, 0.753, 0.776, 0.84]
    )
    print(KernSmooth.bkde(x, gridsize=256, range_x=(np.min(x), np.max(x))))


if __name__ == "__main__":
    main()
