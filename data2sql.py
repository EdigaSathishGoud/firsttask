import os
import glob
import psycopg2
import csv


def insert_csv_data(csv_file, table_name):
    conn = psycopg2.connect(
        dbname='Customer_360',
        user='postgres',
        password='SQL@469',
        host='localhost',
        port='5432'
    )
    cursor = conn.cursor()

    
    with open(csv_file, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        header = next(csv_reader)  # Skip header row
        columns = ', '.join(col.strip('\ufeff') for col in header)  # Exclude the first column
        for row in csv_reader:
            print(row)
            placeholders = ', '.join(['%s'] * len(row))  # Exclude the first column

            # Check for empty strings in timestamp columns
            row_values = [value if value != "" else None for value in row]  

            query = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'
            cursor.execute(query, row_values)

    conn.commit()
    conn.close()

# Function to process all CSV files in a folder
def process_folder(folder_path):
    for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
        table_name = os.path.splitext(os.path.basename(csv_file))[0].lower()  # Use file name as table name
        print("#########Enterted into :",table_name)
        insert_csv_data(csv_file, table_name)

# Process all CSV files in the 'csv_folder' folder
process_folder(r"C:\Users\saediga\Downloads\100records_syntheticdata\testing")
