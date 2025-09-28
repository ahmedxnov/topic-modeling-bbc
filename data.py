import pandas as pd
from multiprocessing import Pool
from os import cpu_count
from preprocessing import preprocess_text
import math

def read_dataset(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

def cpu_info(number_of_documents: int) -> int:
    count = cpu_count() or 1
    chunk_size = max(1, math.ceil(number_of_documents / (count * 2)))
    return count, chunk_size

def parallel_process(articles: pd.Series, c_count, chunk_size) -> list[str]:
    try:
        with Pool(processes=c_count) as pool:
            processed = pool.map(preprocess_text, articles, chunksize=chunk_size)
        return processed
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        return []
    
