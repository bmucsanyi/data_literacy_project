from matplotlib import pyplot as plt
import pandas as pd
import sys

sys.path.insert(0, "../src/")
from test_hypotheses import make_hypothesis_data, run_hypothesis_tests


def make_plot():
    data = pd.read_csv("../dat/data_clean.csv", dtype={5: "object", 16: "object"})
    filtered = make_hypothesis_data(data)

    fig, ax = plt.subplots(21, 1, figsize=(5, 4 * 20))

    for actual_year in range(2000, 2021):
        p_values = run_hypothesis_tests(actual_year, filtered)

        ax[actual_year - 2000].plot(list(range(1910, actual_year)), p_values)
        ax[actual_year - 2000].axhline(0.05)
        ax[actual_year - 2000].set_title(
            f"year: {actual_year}, p_min:{min(p_values):.4f}, p_max:{max(p_values):.4f}"
        )

        ax[actual_year - 2000].set_xlabel("year")
        ax[actual_year - 2000].set_ylabel("p-value")

    fig.tight_layout()

    plt.savefig("../dat/figA1.pdf")
    plt.show()


if __name__ == "__main__":
    make_plot()
