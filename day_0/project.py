from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv('../.env.local')

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
collection_name = "day0_first_system"

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
)

# Create payload index right after creating the collection and before uploading any data to enable filtering.
# If you add it later, HNSW won't rebuild automatically—bump ef_construct (e.g., 100→101) to trigger a safe rebuild.
client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# dimensions: affordability, quality, popularity, innovation
points=[
    models.PointStruct(
        id=1,
        vector=[0.2, 0.9, 0.7, 0.9], # low affordability, high quality, medium popularity, high innovation
        payload={"name": "Iphone", "category": "electronics", "price": 999},
    ),    
    models.PointStruct(
        id=2,
        vector=[0.9, 0.1, 0.2, 0.8], # high affordability, low quality, low popularity, medium innovation
        payload={"name": "Budget Smartphone", "category": "electronics", "price": 299},
    ),
    models.PointStruct(
        id=3,
        vector=[0.4, 0.9, 0.8, 0.3], # low affordability, high quality, high popularity, low innovation
        payload={"name": "Bestselling Novel", "category": "books", "price": 50},
    ),
    models.PointStruct(
        id=4,
        vector=[0.8, 0.3, 0.2, 0.8], # high affordability, low quality, low popularity, high innovation
        payload={"name": "Smart Home Hub", "category": "electronics", "price": 89},
    ),    
    models.PointStruct(
        id=5,
        vector=[0.5, 0.8, 0.6, 0.1], # medium affordability, high quality, medium popularity, low innovation
        payload={"name": "Chair", "category": "furniture", "price": 49},
    ),
            
]

client.upsert(collection_name=collection_name, points=points)

# Define a query vector for "affordable and innovative"
# dimensions: affordability, quality, popularity, innovation
query_vector = [0.2, 0.5, 0.6, 0.1]

# 1. Basic similarity search
basic_results = client.query_points(collection_name, query=query_vector)
formatted = [
    {        
        "score": point.score,
        "name": point.payload["name"],
        "category": point.payload["category"],
        "price": point.payload["price"],
    }
    for point in basic_results.points
]
df = pd.DataFrame(formatted)
df

# 2. Filtered search (only find electronics)
query_vector = [0.2, 0.5, 0.6, 0.1]

filtered_results = client.query_points(
    collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value="electronics"))]
    ),
)
formatted = [
    {        
        "score": point.score,
        "name": point.payload["name"],
        "category": point.payload["category"],
        "price": point.payload["price"],
    }
    for point in filtered_results.points
]
df = pd.DataFrame(formatted)
df