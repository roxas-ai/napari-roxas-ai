
import pandas as pd
import numpy as np
from pathlib import Path
from napari_roxas_ai._preparation._crossdating_handler import merge_crossdating_files

def test_scaling():
    # Create a temporary input file
    input_file = "test_input.txt"
    # Format: Tab-separated, first column is year (becomes index), second is measurement
    data = {
        "Year": [2000, 2001, 2002],
        "Sample1": [1.0, 2.0, 3.0]
    }
    df = pd.DataFrame(data).set_index("Year")
    df.to_csv(input_file, sep="\t")

    # Target file (pre-filled with existing data)
    target_file = "rings_series.txt"
    target_data = {
        "Year": [2000, 2001, 2002],
        "ExistingSample": [1.0, 2.0, 3.0]
    }
    target_df = pd.DataFrame(target_data).set_index("Year")
    target_df.to_csv(target_file, sep="\t")

    # Scaling factor for 1/100 mm -> 10.0
    scaling_factor = 10.0

    print(f"Initial measurement Sample1[2000]: {df.at[2000, 'Sample1']}")
    
    # Run merge_crossdating_files
    merged_df = merge_crossdating_files([input_file], target_file, scaling_factor)
    
    if merged_df is not None:
        print("Merged DataFrame:")
        print(merged_df)
        
        # Check Sample1 at year 2000
        # If the index is string it would be "2000", if it is int it would be 2000.
        # Let's find out.
        print(f"Index type: {type(merged_df.index[0])}")
        val = merged_df.at["2000", "Sample1"]
        existing_val = merged_df.at["2000", "ExistingSample"]
        print(f"Scaled measurement Sample1[2000]: {val}")
        print(f"Existing measurement (should NOT be scaled): {existing_val}")
        
        if val == 10.0 and existing_val == 1.0:
            print("SUCCESS: Scaling applied correctly to new data only.")
        elif val == 10.0 and existing_val == 10.0:
            print("FAILURE: Existing data was also scaled!")
        else:
            print(f"FAILURE: Unexpected results. val={val}, existing_val={existing_val}")
    else:
        print("FAILURE: merged_df is None")

    # Cleanup
    if Path(input_file).exists(): Path(input_file).unlink()
    if Path(target_file).exists(): Path(target_file).unlink()

def test_no_header_scaling():
    # Create a temporary input file WITHOUT header
    input_file = "test_no_header.txt"
    # Format: Tab-separated, first column is year (becomes index), second is measurement
    content = "2000\t1.0\n2001\t2.0\n2002\t3.0\n"
    with open(input_file, "w") as f:
        f.write(content)

    # Target file (initially empty)
    target_file = "rings_series_no_header.txt"
    pd.DataFrame().to_csv(target_file, sep="\t")

    # Scaling factor for 1/100 mm -> 10.0
    scaling_factor = 10.0

    print(f"Initial measurement Sample (year 2001): 2.0")
    
    # Run merge_crossdating_files
    merged_df = merge_crossdating_files([input_file], target_file, scaling_factor)
    
    if merged_df is not None:
        print("Merged DataFrame (No Header Case):")
        print(merged_df)
        
        # Check Sample at year 2001 (note: 2000 was headers!)
        # So 2001 is the first real data point.
        # But wait, 1.0 was also a header!
        try:
            print(f"Indices: {merged_df.index.tolist()}")
            print(f"Columns: {merged_df.columns.tolist()}")
            val = merged_df.at["2001", "1.0"]
            print(f"Scaled measurement at year 2001 (column '1.0'): {val}")
        except Exception as e:
            print(f"Failed to access expected data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("FAILURE: merged_df is None")

    # Cleanup
    if Path(input_file).exists(): Path(input_file).unlink()
    if Path(target_file).exists(): Path(target_file).unlink()

if __name__ == "__main__":
    test_scaling()
    print("\n--- Testing No Header Case ---")
    test_no_header_scaling()
