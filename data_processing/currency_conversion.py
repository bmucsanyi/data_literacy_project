import numpy as np
import pandas as pd
import re
from currency_converter import CurrencyConverter


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
    value, currencies = convert_usd_usd(value, currencies, year)
    value, currencies = convert_eur_usd(value, currencies, year)
    value, currencies = convert_inr_usd(value, currencies, year)
    value, currencies = convert_gbp_usd(value, currencies, year)
    value, currencies = convert_cad_usd(value, currencies, year)
    value, currencies = convert_aud_usd(value, currencies, year)
    value, currencies = convert_sek_usd(value, currencies, year)
    value, currencies = convert_nok_usd(value, currencies, year)
    value, currencies = convert_zar_usd(value, currencies, year)
    value, currencies = convert_dkk_usd(value, currencies, year)
    value, currencies = convert_rub_usd(value, currencies, year)
    value, currencies = convert_cny_usd(value, currencies, year)
    value, currencies = convert_try_usd(value, currencies, year)
    value, currencies = convert_huf_usd(value, currencies, year)
    value, currencies = convert_pln_usd(value, currencies, year)
    value, currencies = convert_jpy_usd(value, currencies, year)

    return value, currencies


def convert_usd_usd(value, currencies, year):
    """Conversion from USD to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    df_usd = pd.read_csv("./inflation_data/usd.csv")
    np_amount = df_usd["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "$":
            target_year = int(year[i]) if int(year[i]) >= 1910 else 1910
            inflation_factor = (
                current_amount / df_usd[df_usd.year == target_year].amount
            )
            currencies[i] = "usd"
            value[i] = value[i] * inflation_factor
    del df_usd

    return value, currencies


def convert_eur_usd(value, currencies, year):
    """Conversion from EUR to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_eur = pd.read_csv("./inflation_data/eur.csv")
    np_amount = df_eur["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "€":
            target_year = int(year[i]) if int(year[i]) >= 1996 else 1996
            inflation_factor = (
                current_amount / df_eur[df_eur.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "EUR", "USD")
            currencies[i] = "usd"

    del df_eur

    return value, currencies


def convert_inr_usd(value, currencies, year):
    """Conversion from INR to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_inr = pd.read_csv("./inflation_data/inr.csv")
    np_amount = df_inr["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "₹":
            target_year = int(year[i]) if int(year[i]) >= 1960 else 1960
            inflation_factor = (
                current_amount / df_inr[df_inr.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "INR", "USD")
            currencies[i] = "usd"

    del df_inr

    return value, currencies


def convert_gbp_usd(value, currencies, year):
    """Conversion from GBP to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_gbp = pd.read_csv("./inflation_data/gbp.csv")
    np_amount = df_gbp["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "£":
            target_year = int(year[i]) if int(year[i]) >= 1910 else 1910
            inflation_factor = (
                current_amount / df_gbp[df_gbp.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "GBP", "USD")
            currencies[i] = "usd"

    del df_gbp

    return value, currencies


def convert_cad_usd(value, currencies, year):
    """Conversion from CAD to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_cad = pd.read_csv("./inflation_data/cad.csv")
    np_amount = df_cad["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "CA$":
            target_year = int(year[i]) if int(year[i]) >= 1915 else 1915
            inflation_factor = (
                current_amount / df_cad[df_cad.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "CAD", "USD")
            currencies[i] = "usd"

    del df_cad

    return value, currencies


def convert_aud_usd(value, currencies, year):
    """Conversion from AUD to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_aud = pd.read_csv("./inflation_data/aud.csv")
    np_amount = df_aud["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "A$":
            target_year = int(year[i]) if int(year[i]) >= 1915 else 1915
            inflation_factor = (
                current_amount / df_aud[df_aud.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "AUD", "USD")
            currencies[i] = "usd"

    del df_aud

    return value, currencies


def convert_sek_usd(value, currencies, year):
    """Conversion from SEK to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_sek = pd.read_csv("./inflation_data/sek.csv")
    np_amount = df_sek["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "SEK":
            target_year = int(year[i]) if int(year[i]) >= 1960 else 1960
            inflation_factor = (
                current_amount / df_sek[df_sek.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "SEK", "USD")
            currencies[i] = "usd"

    del df_sek

    return value, currencies


def convert_nok_usd(value, currencies, year):
    """Conversion from NOK to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_nok = pd.read_csv("./inflation_data/nok.csv")
    np_amount = df_nok["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "NOK":
            target_year = int(year[i]) if int(year[i]) >= 1960 else 1960
            inflation_factor = (
                current_amount / df_nok[df_nok.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "NOK", "USD")
            currencies[i] = "usd"

    del df_nok

    return value, currencies


def convert_zar_usd(value, currencies, year):
    """Conversion from ZAR to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_zar = pd.read_csv("./inflation_data/zar.csv")
    np_amount = df_zar["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "R$":
            target_year = int(year[i]) if int(year[i]) >= 1960 else 1960
            inflation_factor = (
                current_amount / df_zar[df_zar.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "ZAR", "USD")
            currencies[i] = "usd"

    del df_zar

    return value, currencies


def convert_dkk_usd(value, currencies, year):
    """Conversion from DKK to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_dkk = pd.read_csv("./inflation_data/dkk.csv")
    np_amount = df_dkk["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "DKK":
            target_year = int(year[i]) if int(year[i]) >= 1960 else 1960
            inflation_factor = (
                current_amount / df_dkk[df_dkk.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "DKK", "USD")
            currencies[i] = "usd"

    del df_dkk

    return value, currencies


def convert_rub_usd(value, currencies, year):
    """Conversion from RUB to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_rub = pd.read_csv("./inflation_data/rub.csv")
    np_amount = df_rub["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "RUR":
            target_year = int(year[i]) if int(year[i]) >= 1993 else 1993
            inflation_factor = (
                current_amount / df_rub[df_rub.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "RUB", "USD")
            currencies[i] = "usd"

    del df_rub

    return value, currencies


def convert_cny_usd(value, currencies, year):
    """Conversion from NOK to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_cny = pd.read_csv("./inflation_data/cny.csv")
    np_amount = df_cny["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "CN¥":
            target_year = int(year[i]) if int(year[i]) >= 1987 else 1987
            inflation_factor = (
                current_amount / df_cny[df_cny.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "CNY", "USD")
            currencies[i] = "usd"

    del df_cny

    return value, currencies


def convert_try_usd(value, currencies, year):
    """Conversion from NOK to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_try = pd.read_csv("./inflation_data/try.csv")
    np_amount = df_try["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "TRL":
            target_year = int(year[i]) if int(year[i]) >= 1960 else 1960
            inflation_factor = (
                current_amount / df_try[df_try.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "TRY", "USD")
            currencies[i] = "usd"

    del df_try

    return value, currencies


def convert_huf_usd(value, currencies, year):
    """Conversion from HUF to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_huf = pd.read_csv("./inflation_data/huf.csv")
    np_amount = df_huf["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "HUF":
            target_year = int(year[i]) if int(year[i]) >= 1973 else 1973
            inflation_factor = (
                current_amount / df_huf[df_huf.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "HUF", "USD")
            currencies[i] = "usd"

    del df_huf

    return value, currencies


def convert_pln_usd(value, currencies, year):
    """Conversion from PLN to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_pln = pd.read_csv("./inflation_data/jpy.csv")
    np_amount = df_pln["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "PLN":
            target_year = int(year[i]) if int(year[i]) >= 1971 else 1971
            inflation_factor = (
                current_amount / df_pln[df_pln.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "PLN", "USD")
            currencies[i] = "usd"

    del df_pln

    return value, currencies


def convert_jpy_usd(value, currencies, year):
    """Conversion from JPY to USD with corrected inflation of the current year

    input:
        value: nd.array of money value 
        currencies: nd.array with currency symbol
        year: nd.array with year of value 

    output:
        value, currencies with corrected values and symbol 
    """
    c = CurrencyConverter()

    df_jpy = pd.read_csv("./inflation_data/jpy.csv")
    np_amount = df_jpy["amount"].to_numpy()
    current_amount = np_amount[-1]

    for i in range(len(currencies)):
        if currencies[i] == "¥":
            target_year = int(year[i]) if int(year[i]) >= 1960 else 1960
            inflation_factor = (
                current_amount / df_jpy[df_jpy.year == target_year].amount
            )
            value[i] = c.convert(value[i] * inflation_factor, "JPY", "USD")
            currencies[i] = "usd"

    del df_jpy

    return value, currencies

