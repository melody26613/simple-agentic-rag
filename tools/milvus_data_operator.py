import traceback

from pymilvus import connections, Collection, utility
from threading import Lock
from typing import Type, List

from tools.text_embedder import TextEmbedder


class MilvusDataOperator:
    __DB_ALIAS = "data_operator"
    __DB_HOST = "localhost"
    __DB_PORT = "19530"

    def __init__(self, alias=__DB_ALIAS, db_host=__DB_HOST, db_port=__DB_PORT):
        print(f"[MilvusOperator][__init__] {alias=}, {db_host=}, {db_port=}")

        connections.connect(
            alias=alias,
            host=db_host,
            port=db_port,
        )

        self.db_alias = alias
        self.db_host = db_host
        self.db_port = db_port

        self.embedder = TextEmbedder()
        self.mutex = Lock()
        self.collection = None

    def create_collection(self, wrapper_class: Type, collection_name: str) -> bool:
        with self.mutex:
            try:
                wrapper_class(
                    collection_name=collection_name,
                    ip_address=self.db_host,
                    port=self.db_port,
                )

                return True
            except Exception as e:
                print(f"[MilvusOperator][create_collection] {e}")
                print(
                    f"[MilvusOperator][create_collection] exception traceback: {traceback.format_exc()}"
                )
                return False

    def delete_collection(self, collection_name: str) -> bool:
        with self.mutex:
            try:
                collection = Collection(name=collection_name, using=self.db_alias)
                collection.drop()
                return True
            except Exception as e:
                print(f"[MilvusOperator][delete_collection] {e}")
                print(
                    f"[MilvusOperator][delete_collection] exception traceback: {traceback.format_exc()}"
                )
                return False

    def list_collection(self) -> list:
        with self.mutex:
            return utility.list_collections(using=self.db_alias)

    def set_collection(self, collection_name: str) -> bool:
        print(f"[MilvusOperator][set_collection] {collection_name=}")

        with self.mutex:
            try:
                collection = Collection(name=collection_name, using=self.db_alias)
                collection.load()
                self.collection = collection

                return True

            except Exception as e:
                print(f"[MilvusOperator][set_collection] {e}")
                print(
                    f"[MilvusOperator][set_collection] exception traceback: {traceback.format_exc()}"
                )
                return False

    def text_embedding(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)

    def insert_item(self, id: str, item: dict) -> bool:
        if len(id) == 0:
            print(f"[MilvusOperator][insert_item] with invalid id")
            return False
        if self.collection is None:
            print(
                f"[MilvusOperator][insert_item] invalid collection, please call set_collection() before calling this"
            )
            return False

        with self.mutex:
            debug_item = {k: v for k, v in item.items() if "embedding" not in k}
            print(f"[MilvusOperator][insert_item] {id=}, {debug_item=}")

            try:
                collection = self.collection

                item["id"] = id
                collection.insert([item])

                return True

            except Exception as e:
                print(f"[MilvusOperator][insert_item] {e}")
                print(
                    f"[MilvusOperator][insert_item] exception traceback: {traceback.format_exc()}"
                )
                return False

    def delete_item(self, id: str) -> bool:
        if self.collection is None:
            print(
                f"[MilvusOperator][delete_item] invalid collection, please call set_collection() before calling this"
            )
            return False

        with self.mutex:
            print(f"[MilvusOperator][delete_item] {id=}")

            try:
                collection = self.collection

                delete_expression = f"id == '{id}'"
                result = collection.delete(expr=delete_expression)
                print(f"[MilvusOperator][delete_item] {result=}")

                return True

            except Exception as e:
                print(f"[MilvusOperator][delete_item] {e}")
                print(
                    f"[MilvusOperator][delete_item] exception traceback: {traceback.format_exc()}"
                )
                return False

    def list_item(self) -> list:
        if self.collection is None:
            print(
                f"[MilvusOperator][list_item] invalid collection, please call set_collection() before calling this"
            )
            return False

        with self.mutex:
            data = []

            try:
                collection = self.collection
                query = "id != ''"

                iterator = collection.query_iterator(
                    expr=query,
                    output_fields=["*"],
                )

                while True:
                    results = iterator.next()
                    if not results:
                        iterator.close()
                        break

                    data.extend(results)

                return data

            except Exception as e:
                print(f"[MilvusOperator][list_item] {e}")
                print(
                    f"[MilvusOperator][list_item] exception traceback: {traceback.format_exc()}"
                )
                return data

    def flush(self):
        if self.collection is None:
            print(
                f"[MilvusOperator][flush] invalid collection, please call set_collection() before calling this"
            )
            return

        try:
            collection = self.collection
            collection.flush()

        except Exception as e:
            print(f"[MilvusOperator][flush] {e}")
            print(
                f"[MilvusOperator][flush] exception traceback: {traceback.format_exc()}"
            )


if __name__ == "__main__":
    operator = MilvusDataOperator()

    collections = operator.list_collection()
    print(f"type of collections {type(collections)}, {collections=}")
