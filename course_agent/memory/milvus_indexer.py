from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from tools.text_embedder import TextEmbedder, TEXT_EMBEDDING_MODEL_DIM


MAX_NEIGHBOR_SEARCH = 10

TEXT_EMBEDDING_INDEX_PARAMETERS = {
    "IVF_FLAT": {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    },
    "HNSW": {
        "metric_type": "L2",
        "index_type": "HNSW",
        # Adjust M and efConstruction for accuracy/speed trade-off
        "params": {"M": 16, "efConstruction": 200},
    },
}

TEXT_EMBEDDING_SEARCH_PARAMETERS = {
    "IVF_FLAT": {  # default value in langchain Milvus
        "metric_type": "L2",
        "params": {"nprobe": 10},
    },
    "HNSW": {
        "metric_type": "L2",
        "params": {
            "ef": 200,  # Increase ef for better accuracy
        },
    },
}

TEXT_EMBEDDING_INDEX_TYPE = "HNSW"


class MilvusWrapper:
    def __init__(
        self,
        db_name="default",
        collection_name="test",
        ip_address="localhost",
        port=19530,
        drop_collection=False,
        index_type=TEXT_EMBEDDING_INDEX_TYPE,
    ):
        self.collection_name = collection_name
        self.index_type = index_type
        self.collection = self.connect_to_milvus_collection(
            db_name=db_name,
            collection_name=collection_name,
            address=ip_address,
            port=port,
            drop_collection=drop_collection,
        )

    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(
        self,
        db_name,
        collection_name,
        address="localhost",
        port=19530,
        drop_collection=False,
    ):
        connections.connect(db_name=db_name, host=address, port=port)

        if drop_collection:
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                description="ids",
                is_primary=True,
                auto_id=False,
                max_length=1000,
            ),
            FieldSchema(
                name="course_name",
                dtype=DataType.VARCHAR,
                max_length=100,
            ),
            FieldSchema(
                name="course_name_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=TEXT_EMBEDDING_MODEL_DIM,
            ),
            FieldSchema(
                name="course_time",
                dtype=DataType.VARCHAR,
                max_length=100,
            ),
            FieldSchema(
                name="course_time_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=TEXT_EMBEDDING_MODEL_DIM,
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=3000,
            ),
            FieldSchema(
                name="description_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=TEXT_EMBEDDING_MODEL_DIM,
            ),
        ]
        schema = CollectionSchema(fields=fields, description="text search")

        if not utility.has_collection(collection_name):
            collection = Collection(
                name=collection_name, schema=schema
            )  # create collection if not exist
        else:
            collection = Collection(name=collection_name)

        if not collection.has_index(index_name="course_name_embedding"):
            collection.create_index(
                field_name="course_name_embedding",
                index_params=TEXT_EMBEDDING_INDEX_PARAMETERS[self.index_type],
            )
        if not collection.has_index(index_name="course_time_embedding"):
            collection.create_index(
                field_name="course_time_embedding",
                index_params=TEXT_EMBEDDING_INDEX_PARAMETERS[self.index_type],
            )
        if not collection.has_index(index_name="description_embedding"):
            collection.create_index(
                field_name="description_embedding",
                index_params=TEXT_EMBEDDING_INDEX_PARAMETERS[self.index_type],
            )

        return collection


class MilvusSearcher():
    def __init__(
        self,
        db_name="default",
        collection_name="",
        db_ip="localhost",
        db_port=19530,
        index_type=TEXT_EMBEDDING_INDEX_TYPE,
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.index_type = index_type

        self.embedder = TextEmbedder()

        self.reset()

    def reset(self):
        self.milv_wrapper = MilvusWrapper(
            db_name=self.db_name,
            collection_name=self.collection_name,
            ip_address=self.db_ip,
            port=self.db_port,
            index_type=self.index_type,
        )

        course_name_searcher = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.collection_name,
            vector_field="course_name_embedding",
            text_field="course_name",
            search_params=TEXT_EMBEDDING_SEARCH_PARAMETERS[self.index_type],
        )
        self.course_name_retriever = course_name_searcher.as_retriever(
            search_kwargs={"k": MAX_NEIGHBOR_SEARCH}
        )

        course_time_searcher = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.collection_name,
            vector_field="course_time_embedding",
            text_field="course_time",
            search_params=TEXT_EMBEDDING_SEARCH_PARAMETERS[self.index_type],
        )
        self.course_time_retriever = course_time_searcher.as_retriever(
            search_kwargs={"k": MAX_NEIGHBOR_SEARCH}
        )

        description_searcher = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.collection_name,
            vector_field="description_embedding",
            text_field="description",
            search_params=TEXT_EMBEDDING_SEARCH_PARAMETERS[self.index_type],
        )
        self.description_retriever = description_searcher.as_retriever(
            search_kwargs={"k": MAX_NEIGHBOR_SEARCH}
        )

    def __memory_to_string(self, doc_list: list[Document]):
        out_string = ""

        for doc in doc_list:
            course_name = doc.metadata.get("course_name", doc.page_content)
            course_time = doc.metadata.get("course_time", doc.page_content)
            description = doc.metadata.get("description", doc.page_content)

            s = f"""Course '{course_name}' at time '{course_time}' with detailed information: {description}\n\n"""
            out_string += s
        return out_string

    def search_by_course_name(self, query: str) -> str:
        docs = self.course_name_retriever.invoke(query)
        return self.__memory_to_string(docs)

    def search_by_course_time(self, query: str) -> str:
        docs = self.course_time_retriever.invoke(query)
        return self.__memory_to_string(docs)

    def search_by_description(self, query: str) -> str:
        docs = self.description_retriever.invoke(query)
        return self.__memory_to_string(docs)