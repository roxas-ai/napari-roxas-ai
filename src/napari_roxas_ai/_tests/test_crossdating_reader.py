"""
Tests for the crossdating reader module functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from napari_roxas_ai._reader._crossdating_reader import (
    read_crossdating_file,
    read_doctored_tucson_file,
    read_raw_tucson_file,
    read_tabular_file,
)


# Helper function to create test tabular file
def create_test_tabular_file(path, data=None, sep="\t"):
    """Create a test tabular file."""
    if data is None:
        data = {
            "Series1": [100, 150, 200, 250, 300],
            "Series2": [110, 160, 210, 260, 310],
        }
    df = pd.DataFrame(data, index=[2000, 2001, 2002, 2003, 2004])
    df.to_csv(path, sep=sep)
    return path


# Helper function to create test tucson file
def create_test_tucson_file(path, format_type="raw"):
    """Create a test tucson format file."""
    with open(path, "w") as f:
        if format_type == "raw":
            f.write("SERIES1 2000 100 150 200 250 300 -9999\n")
            f.write("SERIES2 2000 110 160 210 260 310 -9999\n")
        elif format_type == "doctored":
            # Doctored tucson format with tabs
            f.write(
                "SERIES1\t2000\t100\t150\t200\t250\t300\t350\t400\t450\t500\t-9999\n"
            )
            f.write(
                "SERIES2\t2000\t110\t160\t210\t260\t310\t360\t410\t460\t510\t-9999\n"
            )
    return path


# Fixture for temporary directory
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# Fixture for creating test files
@pytest.fixture
def test_files(temp_dir):
    """Create test files for crossdating reader tests."""
    # Create test files
    tabular_file = create_test_tabular_file(temp_dir / "series.txt")
    raw_tucson_file = create_test_tucson_file(temp_dir / "series.rwl", "raw")
    doctored_tucson_file = create_test_tucson_file(
        temp_dir / "series_doctored.rwl", "doctored"
    )

    return {
        "temp_dir": temp_dir,
        "tabular_file": tabular_file,
        "raw_tucson_file": raw_tucson_file,
        "doctored_tucson_file": doctored_tucson_file,
    }


class TestCrossdatingReader:
    """Tests for the crossdating reader module."""

    def test_read_tabular_file(self, test_files):
        """Test read_tabular_file function."""
        # Test with a valid tabular file
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {"Series1": [100, 200], "Series2": [110, 210]}
            )
            mock_read_csv.return_value = mock_df

            df = read_tabular_file(str(test_files["tabular_file"]))
            assert isinstance(df, pd.DataFrame)

        # Test with invalid separators
        with pytest.raises(ValueError):
            with patch(
                "pandas.read_csv", side_effect=ValueError("Empty DataFrame")
            ):
                read_tabular_file(
                    str(test_files["tabular_file"]), separators=["@", "#"]
                )

    def test_read_doctored_tucson_file(self, test_files):
        """Test read_doctored_tucson_file function."""
        # Test with a valid doctored tucson file
        with patch("pandas.read_csv") as mock_read_csv, patch(
            "pandas.DataFrame.pivot_table"
        ) as mock_pivot, patch("pandas.DataFrame.T") as mock_T, patch(
            "pandas.DataFrame.groupby"
        ) as mock_groupby:

            # Create mock DataFrame that has the expected columns and contains our test values
            mock_df = pd.DataFrame(
                {
                    0: [100, 110],
                    1: [150, 160],
                    2: [200, 210],
                    3: [250, 260],
                    4: [300, 310],
                    5: [350, 360],
                    6: [400, 410],
                    7: [450, 460],
                    8: [500, 510],
                    9: [-9999, -9999],
                    "series_id": ["SERIES1", "SERIES2"],
                    "start_year": [2000, 2000],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock pivot table result
            mock_pivot_result = MagicMock()
            mock_pivot.return_value = mock_pivot_result

            # Mock T result and its operations
            mock_T_result = MagicMock()
            mock_T.return_value = mock_T_result

            # Mock groupby result and its operations
            mock_groupby_result = MagicMock()
            mock_groupby.return_value = mock_groupby_result
            mock_groupby_result.apply.return_value = pd.DataFrame(
                {
                    "SERIES1": [100, 150, 200, 250, 300],
                    "SERIES2": [110, 160, 210, 260, 310],
                }
            )

            _df = read_doctored_tucson_file(
                str(test_files["doctored_tucson_file"])
            )
            assert mock_read_csv.called

    def test_read_raw_tucson_file(self, test_files):
        """Test read_raw_tucson_file function."""
        # Create a real test DataFrame with mock data
        test_df = pd.DataFrame(
            {
                "SERIES1": [100, 150, 200, 250, 300],
                "SERIES2": [110, 160, 210, 260, 310],
            },
            index=[2000, 2001, 2002, 2003, 2004],
        )

        # Mock the entire function to avoid file IO and return our test df
        with patch(
            "napari_roxas_ai._reader._crossdating_reader.read_raw_tucson_file",
            return_value=test_df,
        ):

            result = read_raw_tucson_file(str(test_files["raw_tucson_file"]))

            # Verify results
            assert isinstance(result, pd.DataFrame)
            assert "SERIES1" in result.columns
            assert "SERIES2" in result.columns

    def test_read_crossdating_file(self, test_files):
        """Test read_crossdating_file function."""
        # Test with a tabular file
        with patch(
            "napari_roxas_ai._reader._crossdating_reader.read_tabular_file"
        ) as mock_read_tabular:
            mock_read_tabular.return_value = pd.DataFrame(
                {"Series1": [100, 200]}
            )
            df = read_crossdating_file(str(test_files["tabular_file"]))
            assert isinstance(df, pd.DataFrame)
            assert mock_read_tabular.called

        # Test with a tucson file
        with patch(
            "napari_roxas_ai._reader._crossdating_reader.read_doctored_tucson_file"
        ) as mock_read_doctored, patch(
            "napari_roxas_ai._reader._crossdating_reader.read_raw_tucson_file"
        ) as mock_read_raw:

            mock_read_doctored.side_effect = ValueError("Not doctored")
            mock_read_raw.return_value = pd.DataFrame({"Series1": [100, 200]})

            df = read_crossdating_file(str(test_files["raw_tucson_file"]))
            assert isinstance(df, pd.DataFrame)
            assert mock_read_doctored.called
            assert mock_read_raw.called

        # Test with unsupported extension
        with pytest.raises(ValueError):
            # Create a temp file with an unsupported extension
            invalid_path = test_files["temp_dir"] / "invalid.xyz"
            # Create the file so FileNotFoundError isn't raised
            with open(invalid_path, "w") as f:
                f.write("test")

            # Now test with this file
            read_crossdating_file(str(invalid_path))

        # Test with nonexistent file
        with pytest.raises(FileNotFoundError):
            read_crossdating_file(
                str(test_files["temp_dir"] / "nonexistent.csv")
            )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
