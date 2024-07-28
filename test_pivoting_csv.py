import pandas as pd

colombo = "data\Colombo S & M Dasboard.csv"
colombo = pd.read_csv(colombo)
print("colombo", colombo.dtypes)



def read_csv_with_float64_to_float32_conversion(file_path):
    # Read the CSV file to inspect column dtypes
    df = pd.read_csv(file_path)

    # Dictionary to store columns to convert from float64 to float32
    convert_to_float32 = {}

    # Iterate over columns to identify float64 columns
    for col_name, dtype in df.dtypes.items():
        if dtype == 'float64':
            convert_to_float32[col_name] = 'float32'

    # Read CSV file again with specified dtype conversions
    df_converted = pd.read_csv(file_path, dtype=convert_to_float32)
    print("colombo_converted", df_converted.dtypes)

    return df_converted

# Example usage:
file_path = 'data\Colombo S & M Dasboard.csv'
df = read_csv_with_float64_to_float32_conversion(file_path)
