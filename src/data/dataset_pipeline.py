import pandas as pd
from multiprocessing import Pool
from os import cpu_count
from src.utils.preprocessing import preprocess_text
import math

def read_dataset(file_path: str, column: str = "Article") -> pd.Series:
    try:
        series = pd.read_csv(file_path, usecols=[column])[column]
        series = series.dropna()
        series = series[series.str.strip() != ""]
        if series.empty:
            raise ValueError(f"CSV '{file_path}' has no usable rows in column '{column}'.")
        return series.reset_index(drop=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV not found: {file_path}") from e
    except ValueError as e:
        raise ValueError(f"Column '{column}' not found in CSV or it is empty.") from e

def cpu_info(number_of_documents: int) -> tuple[int, int]:
    count = cpu_count() or 1
    chunk_size = max(1, math.ceil(number_of_documents / (count * 2)))
    return count, chunk_size

def parallel_preprocessing(articles: pd.Series, c_count: int, chunk_size: int) -> list[list[str]]:
    try:
        with Pool(processes=c_count) as p:
            preprocessed_texts = p.map(preprocess_text, articles, chunksize=chunk_size)
        return preprocessed_texts
    except Exception as e:
        raise RuntimeError(f"Parallel processing failed: {e}") from e
