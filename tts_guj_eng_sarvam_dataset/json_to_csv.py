import json
import csv
import sys
import os

def json_to_csv(json_file, csv_file):
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure JSON is a list of records
    if isinstance(data, dict):
        data = [data]

    if not data:
        print("JSON file is empty.")
        return

    # Get CSV columns from JSON keys
    keys = data[0].keys()

    # Write CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)

        writer.writeheader()
        writer.writerows(data)

    print(f"Converted {json_file} -> {csv_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_csv.py input.json output.csv")
        sys.exit(1)

    json_to_csv(sys.argv[1], sys.argv[2])