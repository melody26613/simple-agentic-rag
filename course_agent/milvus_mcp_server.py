import os
import asyncio

from mcp.server.fastmcp import FastMCP
from course_agent.memory.milvus_indexer import MilvusSearcher

collection = os.getenv("COLLECTION", None)
db_ip = os.getenv("DB_IP", None)
db_port = os.getenv("DB_PORT", None)
db_name = os.getenv("DB_NAME", "default")

if collection is None or len(collection) == 0:
    raise Exception("[Error] invalid environment variable value for 'COLLECTION'")

if db_ip is None or len(db_ip) == 0:
    raise Exception("[Error] invalid environment variable value for 'DB_IP'")

if db_port is None or len(db_port) == 0:
    raise Exception("[Error] invalid environment variable value for 'DB_PORT'")


mcp = FastMCP("milvus_db_for_course")

searcher = MilvusSearcher(
    collection_name=collection,
    db_ip=db_ip,
    db_port=db_port,
    db_name=db_name,
)


@mcp.tool()
async def search_by_course_name(query: str) -> str:
    """
    You can get the most up-to-date computer science course info at National Chung Hsing University.
    Based on the question and your context, decide what text to search for in the database in field course name.
    The query will then search your memories for you.
    """
    return await searcher.search_by_course_name_async(query)


@mcp.tool()
async def search_by_course_time(query: str) -> str:
    """
    You can get the most up-to-date computer science course info at National Chung Hsing University.
    Based on the question and your context, decide what text to search for in the database in field course time.
    The query will then search course details for you.
    """
    return await searcher.search_by_course_time_async(query)


@mcp.tool()
async def search_by_description(query: str) -> str:
    """
    You can get the most up-to-date computer science course info at National Chung Hsing University.
    Based on the question and your context, decide what text to search for in the database in field description, including the course content, like the goal of the course and the detail content that student can learn from this course.
    The query will then search course details for you.
    """
    return await searcher.search_by_description_async(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")

    # result = asyncio.run(search_by_course_name("Android"))
    # print(f"{result=}")

    # result = asyncio.run(search_by_course_time("Fri"))
    # print(f"{result=}")

    # result = asyncio.run(search_by_description("firmware"))
    # print(f"{result=}")
