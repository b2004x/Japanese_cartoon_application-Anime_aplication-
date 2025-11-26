import pandas as pd
import numpy as np
import re
import faiss
from rapidfuzz import process

def load_anime_data(path):
    df = pd.read_csv(path)
    return df


def parse_duration(d):
    if pd.isna(d):   # skip NaN safely
        return np.nan
    
    # make sure it's a string
    d = str(d).lower()
    minutes = 0

    # extract hours (with or without dot, like "1 hr" or "1 hr.")
    hr_match = re.search(r"(\d+)\s*hr\.?", d)
    if hr_match:
        minutes += int(hr_match.group(1)) * 60
    
    # extract minutes (with or without dot, like "24 min" or "24 min.")
    min_match = re.search(r"(\d+)\s*min\.?", d)
    if min_match:
        minutes += int(min_match.group(1))
    
    return minutes if minutes > 0 else np.nan


def preprocess_anime_data(df_anime):
    df_anime = df_anime.replace("Unknown", np.nan)
    df_anime["Episodes"] = pd.to_numeric(df_anime["Episodes"], errors="coerce")
    df_anime["Score"] = pd.to_numeric(df_anime["Score"], errors="coerce")
    df_anime["Duration"] = df_anime["Duration"].apply(parse_duration)
    return df_anime



def preprocess_anime_data_sypnosis(df_anime_sypnosis):
    df_anime_sypnosis = df_anime_sypnosis.replace("Unknown", np.nan)
    return df_anime_sypnosis