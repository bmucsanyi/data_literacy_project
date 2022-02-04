import pandas as pd
from currency_conversion import convert_column

def conversion(df):
    """convert:
            
            -Budget 
            -Gross US & Canada
            -Opening weekend US & Canada
            -Gross worldwide
        
        into USD 
    """
    df = convert_column(df, "Budget")
    print("convert Budget: done")

    df = convert_column(df, "Gross US & Canada")
    print("convert Gross US & Canada: done")

    df = convert_column(df, "Opening weekend US & Canada")
    print("convert Opening weekend US & Canada: done")

    df = convert_column(df, "Gross worldwide")
    print("convert Gross worldwide: done")

    return df

def merge_data():
    """Merge several IMDb pd.DataFrame's with the scraped data."""
    # Read the basic data frame of IMDb
    df = pd.read_csv("../dat/imdb_data/title_basics/data.tsv", sep="\t", dtype={4: 'object', 5: 'object'})
    print("overall: ", len(df.index))

    # Sort out any non-movies (e.g tv-shows)
    df = df[df["titleType"] == "movie"]
    print("Number of movies:\t", len(df.index))

    # Read the review data frame of IMDb
    df_reviews = pd.read_csv("../dat/imdb_data/title_ratings/data.tsv", sep="\t")

    df["tconst"] = df["tconst"].astype(str)
    df_reviews["tconst"] = df_reviews["tconst"].astype(str)

    # inner merge of movies and ratings (movies without any votes are dropped)
    df = df.merge(df_reviews, how="inner", on="tconst")
    print("Number of movies with rating:\t", len(df.index))

    # Free up some memory
    del df_reviews

    # Read in our scraped data
    df_scrape = pd.read_csv("../dat/tconst_scraped_data.csv", dtype={6: 'object', 7: 'object'})
    df_scrape = df_scrape[~df_scrape.duplicated(['tconst'], keep="first")]
    print("Number scraped movies:\t", len(df_scrape))

    ## Hard Coding

    # Change Movie ID "" in the basic DF to the new id ""
    # These are the same movie. The basic IMDb data set has an old (invalid) tconst
    df["tconst"] = df["tconst"].replace(["tt11905872"], "tt4131756")
    df["tconst"] = df["tconst"].replace(["tt4332782"], "tt0246007")
    df["tconst"] = df["tconst"].replace(["tt5072702"], "tt4508986")
    df["tconst"] = df["tconst"].replace(["tt6419536"], "tt4481310")

    df = df[~df.duplicated(['tconst'], keep="first")]

    # Drop Movie 
    # "tt7368158", "tt2437136", "tt2584608", "tt6858500",
    # "tt7375242", "tt7598832", "tt7718552", "tt7728678", "tt7738378"
    # "tt8768374", "tt9828428"
    # because it's no longer available
    # Movie not available (404 Error)
    df = df[df.tconst != "tt7368158"]
    df = df[df.tconst != "tt2437136"]
    df = df[df.tconst != "tt2584608"]
    df = df[df.tconst != "tt6858500"]
    df = df[df.tconst != "tt7375242"]
    df = df[df.tconst != "tt7598832"]
    df = df[df.tconst != "tt7718552"]
    df = df[df.tconst != "tt7728678"]
    df = df[df.tconst != "tt7738378"]
    df = df[df.tconst != "tt8768374"]
    df = df[df.tconst != "tt9828428"]
    print("Number of movies after dropping:\t", df.shape[0])

    # Movies with missing start year in orig data
    # However, have a start year
    df.iloc[147505, 5] = "2012"
    df.iloc[148639, 5] = "2020"
    df.iloc[161518, 5] = "2019"
    df.iloc[161520, 5] = "2020"
    df.iloc[178919, 5] = "2021"
    df.iloc[185090, 5] = "2021"
    df.iloc[254051, 5] = "2019"
    df.iloc[259152, 5] = "2018"
    df.iloc[259650, 5] = "2018"
    df.iloc[271440, 5] = "2018"
    df.iloc[271532, 5] = "2016"
    df.iloc[272545, 5] = "2019"

    df["tconst"] = df["tconst"].astype(str)
    df_scrape["tconst"] = df_scrape["tconst"].astype(str)

    # Merge the data frame and the scraped content
    df = df.merge(df_scrape, how="inner", on="tconst")
    print("Number of movies after merge:\t", df.shape[0])

    # Free up some memory
    del df_scrape

    # Read the review data frame of IMDb
    df_crew = pd.read_csv("../dat/imdb_data/title_crew/data.tsv", sep="\t")

    df["tconst"] = df["tconst"].astype(str)
    df_crew["tconst"] = df_crew["tconst"].astype(str)

    # inner merge of movies and ratings (movies without any votes are dropped)
    df = df.merge(df_crew, how="inner", on="tconst")
    print("Number of movies after crew merge:\t", len(df))

    # Free up some memory
    del df_crew

    # Sort according to tconst
    df = df.sort_values("tconst")

    return df

def merge_convert(file):
    """ Merge all DataFrames and convert currencies."""
    df = merge_data()
    df = conversion(df)

    df.to_csv(file, index=False)
    print("File saved!")