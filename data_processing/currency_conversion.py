import numpy as np
import pandas as pd
import re
from currency_converter import CurrencyConverter

CURRENCY_META = {
    "€": {"start_year": 1996, "id": "EUR"},
    "₹": {"start_year": 1960, "id": "INR"},
    "£": {"start_year": 1910, "id": "GBP"},
    "CA$": {"start_year": 1915, "id": "CAD"},
    "A$": {"start_year": 1923, "id": "AUD"},
    "SEK": {"start_year": 1960, "id": "SEK"},
    "NOK": {"start_year": 1960, "id": "NOK"},
    "R$": {"start_year": 1960, "id": "ZAR"},
    "DKK": {"start_year": 1960, "id": "DKK"},
    "RUR": {"start_year": 1993, "id": "RUB"},
    "CN¥": {"start_year": 1987, "id": "CNY"},
    "TRL": {"start_year": 1960, "id": "TRY"},
    "HUF": {"start_year": 1973, "id": "HUF"},
    "PLN": {"start_year": 1971, "id": "PLN"},
    "¥": {"start_year": 1960, "id": "JPY"},
}


def convert_column(df, column):
    """Conversion from multiple currencies to USD with corrected inflation of the year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    column_data = df[column].to_numpy()
    currencies = np.zeros(len(column_data), dtype="U10")
    value = np.zeros(len(column_data), dtype=int)
    year = df["startYear"].to_numpy()

    for i in range(len(column_data)):
        string = column_data[i]
        if not pd.isna(string):
            value[i] = int("".join(re.findall(r"[\d]+", string)))
            currencies[i] = re.findall(r"[^{\d,\xa0}]+", string)[0]

    value, currencies = convert_all(value, currencies, year)

    df.loc[:, column] = value
    df.loc[currencies != "usd", column] = np.nan

    return df


def convert_all(value, currencies, year):
    """Conversion from multiple currencies to USD with corrected inflation of the year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol
    """
    converter = CurrencyConverter()

    value, currencies = convert_usd_usd(value, currencies, year)
    for currency in CURRENCY_META:
        value, currencies = convert_other_usd(
            currency, CURRENCY_META[currency], currencies, value, year, converter
        )

    return value, currencies


def convert_usd_usd(film_currencies, film_values, film_years):
    """Correct inflation of USD to current year

    input:
        film_currencies: np.ndarray
            column corresponding to the currencies of the films
        film_values: np.ndarray
            column corresponding to the values of the films
        film_years: pd.Series
            column corresponding to the startYear of the films
    returns:
        value, currencies with corrected values and symbol 
    """
    df_usd = pd.read_csv("./inflation_data/usd.csv")
    np_amount = df_usd["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(film_currencies)):
        if film_currencies[i] == "$":
            target_year = int(film_years[i]) if int(film_years[i]) >= 1910 else 1910
            inflation_factor = (
                current_amount / df_usd[df_usd.year == target_year].amount
            )
            film_currencies[i] = "usd"
            film_values[i] = film_values[i] * inflation_factor

    return film_values, film_currencies


def convert_other_usd(
    currency, currency_dict, film_currencies, film_values, film_years, converter
):
    """Conversion from ``currency`` to USD with corrected inflation of the current year

    input:
        currency: str
            the currency we wish to convert to USD
        currency_dict: dict
            dictionary containing lowest conversion year and id
        film_currencies: np.ndarray
            column corresponding to the currencies of the films
        film_values: np.ndarray
            column corresponding to the values of the films
        film_years: pd.Series
            column corresponding to the startYear of the films
        converter: CurrencyConverter

    returns:
        value, currencies with corrected values and symbol 
    """
    currency_id = currency_dict["id"]
    df = pd.read_csv(f"./inflation_data/{currency_id.lower()}.csv")
    np_amount = df["amount"].to_numpy()
    current_amount = np_amount[-1]
    start_year = currency_dict["start_year"]

    for i in range(len(film_currencies)):
        if film_currencies[i] == currency:
            target_year = (
                int(film_years[i]) if int(film_years[i]) >= start_year else start_year
            )
            inflation_factor = current_amount / df[df.year == target_year].amount
            film_values[i] = converter.convert(
                film_values[i] * inflation_factor, currency_id, "USD"
            )
            film_currencies[i] = "usd"

    return film_values, film_currencies
