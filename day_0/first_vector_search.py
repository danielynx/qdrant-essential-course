from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv('../.env.local')

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Define the collection name
collection_name = "my_first_collection"

# Create the collection with specified vector parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=4,  # Dimensionality of the vectors
        distance=models.Distance.COSINE  # Distance metric for similarity search
    )
)

# Retrieve and display the list of collections
collections = client.get_collections()
print("Existing collections:", collections)

# Define the vectors to be inserted
points = [
    models.PointStruct(
        id=1,
        vector=[0.1, 0.2, 0.3, 0.4],  # 4D vector
        payload={"category": "example"}  # Metadata (optional)
    ),
    models.PointStruct(
        id=2,
        vector=[0.2, 0.3, 0.4, 0.5],
        payload={"category": "demo"}
    )
]

# Insert vectors into the collection
client.upsert(
    collection_name=collection_name,
    points=points
)

collection_info = client.get_collection(collection_name)
print("Collection info:", collection_info)

# Similarity search
query_vector = [0.08, 0.14, 0.33, 0.28]

search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=1  # Return the top 1 most similar vector
)

print("Search results:", search_results)

# Filter search
client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

search_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="category",
            match=models.MatchValue(value="demo")
        )
    ]
)

filtered_search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=1,
    query_filter=search_filter
)

print("Filtered search results:", filtered_search_results)