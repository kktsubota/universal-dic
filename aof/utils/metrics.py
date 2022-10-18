import numpy as np
import scipy.interpolate


def BD_RATE(
    R1: np.ndarray,
    PSNR1: np.ndarray,
    R2: np.ndarray,
    PSNR2: np.ndarray,
    piecewise: int = 0,
) -> np.float64:
    """calculate the BD-rate
    NO LISENCE code.
    https://github.com/Anserw/Bjontegaard_metric

    $ wget https://raw.githubusercontent.com/Anserw/Bjontegaard_metric/master/bjontegaard_metric.py -O utils.py
    """
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(
            np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples
        )
        v2 = scipy.interpolate.pchip_interpolate(
            np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples
        )
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff
