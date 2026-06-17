"""
xCAD Table Loader

Loads 7 raw CSV files from Alibaba Cluster Trace GPU v2020.
Applies schema column names, basic cleaning, and memory-optimized dtypes.

Usage:
    from src.utils.load_tables import load_all_tables
    tables = load_all_tables()
"""

import os
import logging
from typing import Dict, Optional
import pandas as pd
from src.utils.schema import (
    TABLE_FILES,
    TABLE_COLS,
    TABLE_DTYPES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_single_table(
    table_name: str,
    chunked: bool = False,
    chunksize: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a single CSV file with column names from schema.

    Args:
        table_name: Key in TABLE_FILES (e.g., 'pai_job_table')
        chunked: If True, return iterator instead of DataFrame
        chunksize: Chunk size for chunked reading

    Returns:
        DataFrame with correct column names and types
    """
    if table_name not in TABLE_FILES:
        raise ValueError(f"Unknown table: {table_name}")

    file_path = TABLE_FILES[table_name]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    col_names = TABLE_COLS[table_name]
    col_count = len(col_names)

    logger.info(f"Loading {table_name} from {file_path}")

    def read_with_fallback():
        try:
            return pd.read_csv(
                file_path,
                header=None,
                names=col_names,
                dtype=TABLE_DTYPES.get(table_name),
                na_values=["", "NA", "N/A", "null", "NULL"],
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            logger.warning(f"Standard read failed: {e}, retrying with c engine")
            return pd.read_csv(
                file_path,
                header=None,
                names=col_names,
                dtype=TABLE_DTYPES.get(table_name),
                na_values=["", "NA", "N/A", "null", "NULL"],
                engine="c",
                on_bad_lines="skip",
            )

    if chunked:
        return pd.read_csv(
            file_path,
            header=None,
            names=col_names,
            dtype=TABLE_DTYPES.get(table_name),
            chunksize=chunksize,
            na_values=["", "NA", "N/A", "null", "NULL"],
            engine="python",
            on_bad_lines="skip",
        )

    df = read_with_fallback()

    initial_rows = len(df)
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    cleaned_rows = len(df)

    logger.info(
        f"  Rows: {cleaned_rows} ({initial_rows - cleaned_rows} dropped as empty/duplicate)"
    )

    return df


def load_all_tables(
    sample_size: Optional[int] = None,
    chunked_threshold_mb: float = 500.0
) -> Dict[str, pd.DataFrame]:
    """
    Load all 7 tables.

    Args:
        sample_size: If set, only load first N rows (for testing)
        chunked_threshold_mb: Files larger than this use chunked reading

    Returns:
        Dict mapping table_name → DataFrame
    """
    tables = {}

    for table_name in TABLE_FILES:
        file_path = TABLE_FILES[table_name]
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        try:
            if sample_size:
                df = load_single_table(table_name)
                df = df.head(sample_size)
                logger.info(f"  Sampled to {len(df)} rows")
            elif file_size_mb > chunked_threshold_mb:
                logger.info(f"  Large file ({file_size_mb:.1f}MB), loading in chunks")
                chunks = []
                for chunk in load_single_table(table_name, chunked=True, chunksize=100000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = load_single_table(table_name)

            tables[table_name] = df
            logger.info(f"  {table_name}: {len(df)} rows, {len(df.columns)} cols, "
                       f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")

        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            raise

    return tables


def load_test(output_path: str = None):
    """
    Run minimal test: print row count, column names, first 2 rows for each table.
    Write results to load_test.log.
    """
    from src.utils.config import OUTPUT_ROOT
    if output_path is None:
        output_path = f"{OUTPUT_ROOT}/logs/load_test.log"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("xCAD Load Test Results\n")
        f.write("=" * 80 + "\n\n")

        try:
            tables = load_all_tables()

            for table_name, df in tables.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"Table: {table_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Rows: {len(df)}\n")
                f.write(f"Columns: {list(df.columns)}\n")
                f.write(f"\nFirst 2 rows:\n")
                f.write(df.head(2).to_string())
                f.write("\n\n")

                logger.info(f"{table_name}: {len(df)} rows, {len(df.columns)} cols")

        except Exception as e:
            error_msg = f"Load test failed: {e}"
            f.write(error_msg + "\n")
            logger.error(error_msg)
            raise

    logger.info(f"Load test complete. Results written to {output_path}")
    return output_path


if __name__ == "__main__":
    load_test()
