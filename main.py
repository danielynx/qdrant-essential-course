def main():
    print("Hello from qdrant-essential-course!")


if __name__ == "__main__":
    main()

from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://119a4b2c-dce2-436e-8f83-389add3e7fb2.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.XdlQ0UjxYj-Db1RZEV9zG_S8rqVgFAw33eeoAwx8z_M",
)

print(qdrant_client.get_collections())