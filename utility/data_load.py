
import pandas as pd

def load_USGS(filepath: str = 'data/USGS_data.csv') -> pd.DataFrame:
    """Loads USGS dataset from filepath

    Args:
        filepath (str): Path to the USGS dataset CSV file.
    """
    df = pd.read_csv(filepath)
    df.index = pd.to_datetime(df['DateTime'])
    df.drop(columns=['DateTime'], inplace=True)
    return df

def load_NLRDB(filepath: str ='data/NLRDB_combined.csv') -> pd.DataFrame:
    """Loads NLRDB dataset from filepath

    Args:
        filepath (str): Path to the NLRDB dataset CSV file.
    """
    df2 = pd.read_csv(filepath)
    df2.index = pd.to_datetime(df2['DateTime'])
    return df2

def load_dataset(USGS:str = 'data/USGS_data.csv', NLRDB:str = 'data/NLRDB_combined.csv', start:str = '2020-01-01', end:str = '2025-12-31') -> pd.DataFrame: 
    """
    Load the USGS and NLRDB datasets from CSV files.

    Parameters:
    USGS (str): Path to the USGS dataset CSV file.
    NLRDB (str): Path to the NLRDB dataset CSV file.

    Returns:
    dataframe: A combined dataframe containing both USGS and NLRDB datasets.
    """
    df_USGS = load_USGS(USGS)
    df_NLRDB = load_NLRDB(NLRDB)
    df_combined = pd.concat([df_USGS, df_NLRDB], axis=1)
    df_combined = df_combined.loc[start:end]
    df_combined.drop(columns="DateTime", inplace=True)
    return df_combined