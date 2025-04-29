"""
Helper functions to read crossdating data files.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def read_crossdating_file(path: str) -> pd.DataFrame:
    """
    Read a crossdating file and return it as a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the crossdating file

    Returns
    -------
    pd.DataFrame
        DataFrame containing the crossdating data.
        The DataFrame has tree-ring series as columns (or index depending on format),
        and years as the other dimension.

    Raises
    ------
    ValueError
        If the file format is not supported or if the file cannot be read
    FileNotFoundError
        If the file does not exist
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Get file extension
    file_ext = Path(path).suffix.lower()

    # Process based on file extension
    if file_ext in [".csv", ".tsv", ".txt"]:
        try:
            df = read_tabular_file(path)
            return df
        except ValueError as e:
            raise ValueError(f"Failed to read tabular file: {str(e)}") from e

    elif file_ext in [".rwl", ".tuc"]:
        # Try doctored tucson format first, then raw tucson if that fails
        try:
            df = read_doctored_tucson_file(path)
            return df
        except ValueError:
            try:
                df = read_raw_tucson_file(path)
                return df
            except (ValueError, OSError, UnicodeDecodeError) as e:
                raise ValueError(
                    f"Failed to read Tucson file: {str(e)}"
                ) from e
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def read_tabular_file(
    path: str, separators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read a tabular file and return it as a pandas DataFrame.
    Parameters
    ----------
    path : str
        Path to the tabular file
    separators : Optional[List[str]]
        List of separators to try when reading the file
    Returns
    -------
    pd.DataFrame
        DataFrame containing the tabular data
    Raises
    -------
    ValueError
        If the file cannot be read with any of the provided separators
    """
    if separators is None:
        separators = ["\t", ",", ";"]

    for sep in separators:
        df = pd.read_csv(path, sep=sep, index_col=0)
        if len(df.columns) > 0:
            return df

    raise ValueError(
        f"Could not read the file with any of the provided separators: {separators}"
    )


def read_doctored_tucson_file(
    path: str, end_of_line_values: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Read a doctored Tucson file (separated by tabs), and return it as a pandas DataFrame.
    Parameters
    ----------
    path : str
        Path to the doctored Tucson file
    end_of_line_values : Optional[List[int]]
        List of values that indicate the end of a line in the file
    Returns
    -------
    pd.DataFrame
        DataFrame containing the doctored Tucson data
    """
    if end_of_line_values is None:
        end_of_line_values = [-9999, -999, 9999, 999]

    # Read the file with tab separator
    df = pd.read_csv(path, sep="\t", header=None)

    # Check if the DataFrame has more than one column
    if len(df.columns) > 1:
        # Get end of line value
        end_of_line_value = None
        for value in end_of_line_values:
            if value in df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values:
                end_of_line_value = value
                break

        if end_of_line_value is None:
            raise ValueError("Could not detect end-of-line value in the file")

        # Change column names
        df.columns = ["series_id", "start_year", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Replace end of line values with NaN
        df.replace(end_of_line_value, np.nan, inplace=True)
        # Pivot the DataFrame
        df = df.pivot_table(
            index="series_id",
            columns="start_year",
            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        # Assign year values as columns
        df.columns = [sum(a) for a in df.columns.to_list()]
        # Merge duplicate columns
        df = df.T.groupby(level=0).apply(
            lambda group: group.bfill(axis=0).iloc[0, :]
        )

        return df
    else:
        # If the DataFrame has only one column, return it as is
        raise ValueError("The file is not a tab-delimited tucson file.")


def read_raw_tucson_file(
    path: str, end_of_line_values: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Read a raw Tucson file and return it as a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the raw Tucson file
    end_of_line_values : Optional[List[int]]
        List of values that indicate the end of a series

    Returns
    -------
    pd.DataFrame
        DataFrame containing the tree-ring data

    Raises
    ------
    ValueError
        If the file format is invalid or cannot be parsed
    """
    if end_of_line_values is None:
        end_of_line_values = [-9999, -999, 9999, 999]

    try:
        # Get end of line value
        end_of_line_value = None
        with open(path) as file:
            data = file.read()
            for value in end_of_line_values:
                if str(value) in data:
                    end_of_line_value = value
                    break

        if end_of_line_value is None:
            raise ValueError("Could not detect end-of-line value in the file")

        # Read the file
        series_dict = {}  # Use a dictionary to collect data for each series

        with open(path) as file:
            for line_num, line in enumerate(file, 1):
                try:
                    line_parts = line.strip().split()

                    if len(line_parts) <= 2:
                        continue

                    series_id = line_parts[0]
                    try:
                        from_year = int(line_parts[1])
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid year value at line {line_num}: {line_parts[1]}"
                        ) from e

                    # Check for end of line marker
                    if int(line_parts[-1]) == end_of_line_value:
                        line_parts.pop(-1)

                    to_year = from_year + len(line_parts) - 2

                    try:
                        values = [int(val) for val in line_parts[2:]]
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid ring width value at line {line_num}"
                        ) from e

                    # Create a Series for the current line's data
                    current_line_series = pd.Series(
                        data=values, index=range(from_year, to_year)
                    )

                    # If this is a new series, initialize it in our dict
                    if series_id not in series_dict:
                        series_dict[series_id] = current_line_series
                    else:
                        # For existing series, update with new values
                        # This automatically handles the concatenation
                        series_dict[series_id] = series_dict[
                            series_id
                        ].combine_first(current_line_series)

                except ValueError as e:
                    raise ValueError(
                        f"Error processing line {line_num}: {str(e)}"
                    ) from e

        # Convert all series to a DataFrame - addressing the FutureWarning
        if series_dict:
            # Create DataFrame only if we have non-empty data
            # Filter out any empty series to avoid the FutureWarning
            non_empty_series = {
                k: v for k, v in series_dict.items() if not v.empty
            }

            if non_empty_series:
                df = pd.DataFrame(non_empty_series)
            else:
                df = (
                    pd.DataFrame()
                )  # Create an empty DataFrame if all series are empty
        else:
            df = pd.DataFrame()  # Create an empty DataFrame if no series

        if df.empty:
            raise ValueError("No valid data found in the file")

        return df

    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Error reading file: {str(e)}") from e
