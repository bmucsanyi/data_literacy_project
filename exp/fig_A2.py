from matplotlib import pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy.stats import betabinom
import os

sys.path.insert(0, "../src/")
from test_hypotheses import make_hypothesis_data, run_hypothesis_test


def make_plot():
    data = pd.read_csv("../dat/data_clean.csv", dtype={5: "object", 16: "object"})
    filtered = make_hypothesis_data(data)
    threshold = 5
    bad_filtered = filtered[filtered["averageRating"] < threshold]
    good_filtered = filtered[filtered["averageRating"] >= threshold]

    p_valueg, n_1g, m_0g, n_0g = run_hypothesis_test(1910, 2020, good_filtered)
    print("High-rated films p-value", p_valueg)
    p_valueb, n_1b, m_0b, n_0b = run_hypothesis_test(1910, 2020, bad_filtered)
    print("Low-rated films p-value", p_valueb)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Comparison of tendencies of high-rated and low-rated movies")

    pg = betabinom(n_1g, m_0g + 1, n_0g - m_0g + 1)
    mm_1g = np.arange(n_1g)

    ax[0].set_title(f"Rating < 5: $n_1 = {n_1g}, m_0 = {m_0g}, n_0 = {n_0g}$")
    ax[0].vlines(mm_1g, 0, pg.pmf(mm_1g))
    ax[0].axvline(
        len(
            good_filtered[
                (good_filtered["startYear"] == 2020)
                & (good_filtered["Budget"] > good_filtered["Gross worldwide"])
            ]
        )
    )

    pb = betabinom(n_1b, m_0b + 1, n_0b - m_0b + 1)
    mm_1b = np.arange(n_1b)

    ax[1].set_title(f"Rating >= 5: $n_1 = {n_1b}, m_0 = {m_0b}, n_0 = {n_0b}$")
    ax[1].vlines(mm_1b, 0, pb.pmf(mm_1b))
    ax[1].axvline(
        len(
            bad_filtered[
                (bad_filtered["startYear"] == 2020)
                & (bad_filtered["Budget"] > bad_filtered["Gross worldwide"])
            ]
        )
    )

    fig.tight_layout()

    os.makedirs("../doc/gfx", exist_ok=True)
    plt.savefig("../doc/gfx/figA2.pdf")
    plt.show()


if __name__ == "__main__":
    make_plot()
