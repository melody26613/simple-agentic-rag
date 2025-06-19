import pandas as pd
import argparse

from course_agent.memory.milvus_data_operator_course import MilvusDataOperatorCourse

COLLECTION_NAME = "course_collection"

EXCEL_FILEPATH = "course_agent/university_cs_courses.xlsx"


parser = argparse.ArgumentParser()
parser.add_argument("--db_host", type=str, default="localhost")
parser.add_argument("--db_port", type=str, default="19530")

args = parser.parse_args()
print(f"args: {args}")


operator = MilvusDataOperatorCourse(
    db_host=args.db_host,
    db_port=args.db_port,
)


collections = operator.list_collection()
if COLLECTION_NAME in collections:
    print(f"Try to delete collection with name {COLLECTION_NAME}")
    if not operator.delete_collection(COLLECTION_NAME):
        print(f"Cannot delete collection with name {COLLECTION_NAME}")
        exit(0)

if not operator.create_collection(COLLECTION_NAME):
    print(f"Cannot create collection with name {COLLECTION_NAME}")
    exit(0)

operator.set_collection(collection_name=COLLECTION_NAME)


def load_excel_as_iterable(filepath):
    """
    Load an Excel file and convert all sheets into iterable objects (list of dicts).

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        dict: Dictionary with sheet names as keys and list of row dictionaries as values.
    """
    # Load all sheets into a dictionary of DataFrames
    excel_data = pd.read_excel(filepath, sheet_name=None)

    # Convert each DataFrame into a list of row dictionaries
    iterable_data = {
        sheet_name: sheet_df.where(pd.notnull(sheet_df), None).to_dict(orient="records")
        for sheet_name, sheet_df in excel_data.items()
    }

    return iterable_data


excel_iterable = load_excel_as_iterable(EXCEL_FILEPATH)

for sheet_name, rows in excel_iterable.items():
    print(f"Sheet: {sheet_name}")
    for row in rows:
        print(row)
        id = row["id"]

        if not id:
            continue

        item = {
            "course_name": row["course_name"],
            "course_time": row["course_time"],
            "description": row["description"],
        }

        operator.insert_item(
            id=str(id),
            item=item,
        )

operator.flush()

print("finished")
