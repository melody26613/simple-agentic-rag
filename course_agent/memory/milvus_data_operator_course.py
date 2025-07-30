from typing import List

from tools.milvus_data_operator import MilvusDataOperator
from course_agent.memory.milvus_indexer import MilvusWrapper


class MilvusDataOperatorCourse(MilvusDataOperator):
    __DB_ALIAS = "data_operator"
    __DB_HOST = "localhost"
    __DB_PORT = "19530"

    def __init__(self, alias=__DB_ALIAS, db_host=__DB_HOST, db_port=__DB_PORT):
        super().__init__(alias=alias, db_host=db_host, db_port=db_port)

    def create_collection(self, collection_name: str) -> bool:
        return super().create_collection(
            wrapper_class=MilvusWrapper,
            collection_name=collection_name,
        )

    def insert_item(self, id: str, item: dict) -> bool:
        memory_dict = {
            "course_name": item["course_name"],
            "course_name_embedding": super().text_embedding(item["course_name"]),
            "course_time": item["course_time"],
            "course_time_embedding": super().text_embedding(item["course_time"]),
            "description": item["description"],
            "description_embedding": super().text_embedding(item["description"]),
        }

        return super().insert_item(
            id=id,
            item=memory_dict,
        )

    def list_item(self) -> List[dict]:
        data = super().list_item()

        for elem in data:
            reordered = {
                "id": elem["id"],
                "course_name": elem["course_name"],
                "course_time": elem["course_time"],
                "description": elem["description"],
            }
            elem.clear()
            elem.update(reordered)

        return data


if __name__ == "__main__":
    collection_name = "course_collection"
    operator = MilvusDataOperatorCourse()

    # operator.create_collection(collection_name=collection_name)
    # operator.delete_collection(collection_name=collection_name)

    # list collection
    collections = operator.list_collection()
    print(f"type of collections {type(collections)}, {collections=}")

    operator.set_collection(collection_name=collection_name)

    exit()

    # insert item
    memory_item = {
        "course_name": "Class1",
        "course_time": "Wed. 09:00~12:00",
        "description": "Test by Melody Wang.",
    }
    operator.insert_item(id="123", item=memory_item)
    operator.flush()

    # list items
    data = operator.list_item()
    for row in data:
        print(row)

    # delete item
    operator.delete_item(id="123")
    operator.flush()
