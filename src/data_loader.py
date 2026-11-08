import pandas as pd 
from pathlib import Path

def load_data(filename): 

    file_path = Path("data/raw") / filename
    df = pd.read_csv(file_path)

    return df 

