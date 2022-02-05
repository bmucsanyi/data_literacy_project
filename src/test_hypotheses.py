from scipy.stats import betabinom
import pandas as pd


def _clean_unknowns(row, column):
    if row[column] == "\\N":
        return None
    else:
        return row[column]


def make_hypothesis_data(full_dataset):
    full_dataset["startYear"] = full_dataset.apply(
        lambda row: _clean_unknowns(row, "startYear"), axis=1
    )
    full_dataset["startYear"] = pd.to_numeric(full_dataset["startYear"])

    filtered = full_dataset[
        ["startYear", "Budget", "Gross worldwide", "averageRating"]
    ].dropna()

    return filtered


def run_hypothesis_test(earliest_year, actual_year, dataset):
    n_1 = len(dataset[dataset["startYear"] == actual_year])
    m_0 = len(
        dataset[
            (dataset["startYear"].between(earliest_year, actual_year - 1))
            & (dataset["Budget"] > dataset["Gross worldwide"])
        ]
    )
    n_0 = len(dataset[dataset["startYear"].between(earliest_year, actual_year - 1)])
    p_value = 1 - betabinom.cdf(
        len(
            dataset[
                (dataset["startYear"] == actual_year)
                & (dataset["Budget"] > dataset["Gross worldwide"])
            ]
        )
        - 1,
        n_1,
        m_0 + 1,
        n_0 - m_0 + 1,
    )

    return p_value, n_1, m_0, n_0


def run_hypothesis_tests(actual_year, dataset):
    p_values = []
    for earliest_year in range(1910, actual_year):
        p_value, _, _, _ = run_hypothesis_test(earliest_year, actual_year, dataset)
        p_values.append(p_value)

    return p_values
