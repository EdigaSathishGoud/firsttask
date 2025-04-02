import pandas as pd

# Load the Excel file
file_path = r"C:\Users\saediga\Downloads\adding_new_row.xlsx" # Update with your actual file path
df = pd.read_excel(file_path)

# Expand rows where 'value' column has multiple values
expanded_rows = []
for _, row in df.iterrows():
    values = str(row["value"]).split(",")  # Split values by comma
    for val in values:
        expanded_rows.append({"Table_name": row["Table_name"], "value": val.strip()})

# Create a new DataFrame and save it back to Excel
new_df = pd.DataFrame(expanded_rows)
new_file_path = "updated_file.xlsx"
new_df.to_excel(new_file_path, index=False)

print(f"Updated file saved as {new_file_path}")
