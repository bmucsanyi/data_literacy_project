from scipy.stats import betabinom
from matplotlib import pyplot as plt
import pandas as pd


def make_plot():

    data = pd.read_csv("../dat/data_clean.csv", dtype={5: 'object', 16: 'object'})

    def clean_unknowns(row, column):
        if row[column] == "\\N":
            return None
        else:
            return row[column]

    data["startYear"] = data.apply(lambda row: clean_unknowns(row, "startYear"), axis=1)
    data["startYear"] = pd.to_numeric(data["startYear"])

    filtered = data[["startYear", "Budget", "Gross worldwide", "averageRating"]].dropna()
    
    fig, ax = plt.subplots(20,1, figsize=(5,4*20))

    for actual_year in range(2000, 2020):
        p_values = []
        for lower in range(1910, actual_year):
            n_1 = len(filtered[filtered["startYear"] == actual_year])
            m_0 = len(filtered[(filtered["startYear"].between(lower, actual_year - 1)) & (filtered["Budget"] > filtered["Gross worldwide"])])
            n_0 = len(filtered[filtered["startYear"].between(lower, actual_year - 1)])
            p_values.append(1 - betabinom.cdf(len(filtered[(filtered["startYear"] == actual_year) & (filtered["Budget"] > filtered["Gross worldwide"])]) - 1, n_1, m_0 + 1, n_0 - m_0 + 1))

        ax[actual_year-2000].plot(list(range(1910, actual_year)), p_values)
        ax[actual_year-2000].axhline(0.05)
        ax[actual_year-2000].set_title(f"year: {actual_year}, p_min:{min(p_values):.4f}, p_max:{max(p_values):.4f}")

        ax[actual_year-2000].set_xlabel("year")
        ax[actual_year-2000].set_ylabel("p-value")

    fig.tight_layout()
    
    plt.savefig("../dat/figA1.pdf")
    plt.show()


if __name__ == "__main__":
    make_plot()
