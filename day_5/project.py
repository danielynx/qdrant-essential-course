from dotenv import load_dotenv
from datetime import datetime
from qdrant_client import QdrantClient, models
import os

load_dotenv('../.env.local')

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Define the collection name
collection_name = "day5_recommendations_hybrid"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        # Dense vectors for semantic understanding
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        # ColBERT multivectors for fine-grained reranking
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(
                m=0  # Disable HNSW - used only for reranking
            ),
        ),
    },
    sparse_vectors_config={
        # Sparse vectors for exact keyword matching
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False)
        )
    },
)

# Business metadata indexes
client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema="keyword",
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="user_segment",
    field_schema="keyword",
)

# Quality and recency indexes
client.create_payload_index(
    collection_name=collection_name,
    field_name="release_date",
    field_schema="datetime",
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="popularity_score",
    field_schema="float",
)
client.create_payload_index(
    collection_name=collection_name,
    field_name="rating",
    field_schema="float",
)

sample_data = [
    {
        "title": "The Matrix",
        "description": "A hacker discovers reality is a simulation and joins a rebellion against machines.",
        "category": "movie",
        "genre": ["sci-fi", "action"],
        "year": 1999,
        "rating": 8.7,
        "user_segment": "premium",
        "popularity_score": 0.95,
        "release_date": "1999-03-31T00:00:00Z",
    },
    {
        "title": "Inception",
        "description": "A skilled thief steals corporate secrets through dream-sharing technology.",
        "category": "movie",
        "genre": ["sci-fi", "thriller"],
        "year": 2010,
        "rating": 8.8,
        "user_segment": "premium",
        "popularity_score": 0.93,
        "release_date": "2010-07-16T00:00:00Z",
    },
    {
        "title": "Interstellar",
        "description": "Explorers travel through a wormhole in space to ensure humanity's survival.",
        "category": "movie",
        "genre": ["sci-fi", "drama"],
        "year": 2014,
        "rating": 8.6,
        "user_segment": "premium",
        "popularity_score": 0.91,
        "release_date": "2014-11-07T00:00:00Z",
    },
    {
        "title": "The Office",
        "description": "A mockumentary about office workers dealing with daily corporate life.",
        "category": "tv_show",
        "genre": ["comedy"],
        "year": 2005,
        "rating": 8.9,
        "user_segment": "standard",
        "popularity_score": 0.88,
        "release_date": "2005-03-24T00:00:00Z",
    },
    {
        "title": "Breaking Bad",
        "description": "A chemistry teacher turns into a methamphetamine producer.",
        "category": "tv_show",
        "genre": ["crime", "drama"],
        "year": 2008,
        "rating": 9.5,
        "user_segment": "premium",
        "popularity_score": 0.97,
        "release_date": "2008-01-20T00:00:00Z",
    },
    {
        "title": "Stranger Things",
        "description": "Kids uncover supernatural mysteries in a small town.",
        "category": "tv_show",
        "genre": ["sci-fi", "horror"],
        "year": 2016,
        "rating": 8.7,
        "user_segment": "standard",
        "popularity_score": 0.92,
        "release_date": "2016-07-15T00:00:00Z",
    },
    {
        "title": "The Dark Knight",
        "description": "Batman faces the Joker in a battle for Gotham's soul.",
        "category": "movie",
        "genre": ["action", "crime"],
        "year": 2008,
        "rating": 9.0,
        "user_segment": "premium",
        "popularity_score": 0.96,
        "release_date": "2008-07-18T00:00:00Z",
    },
    {
        "title": "The Social Network",
        "description": "The story of the founding of Facebook and its legal battles.",
        "category": "movie",
        "genre": ["drama"],
        "year": 2010,
        "rating": 7.7,
        "user_segment": "standard",
        "popularity_score": 0.80,
        "release_date": "2010-10-01T00:00:00Z",
    },
    {
        "title": "The Mandalorian",
        "description": "A bounty hunter travels across the galaxy protecting a mysterious child.",
        "category": "tv_show",
        "genre": ["sci-fi", "adventure"],
        "year": 2019,
        "rating": 8.8,
        "user_segment": "premium",
        "popularity_score": 0.90,
        "release_date": "2019-11-12T00:00:00Z",
    },
    {
        "title": "Parasite",
        "description": "A poor family schemes to become employed by a wealthy household.",
        "category": "movie",
        "genre": ["thriller", "drama"],
        "year": 2019,
        "rating": 8.6,
        "user_segment": "standard",
        "popularity_score": 0.89,
        "release_date": "2019-05-30T00:00:00Z",
    },
]

texts = [it["description"] for it in sample_data]

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

# Model configurations
DENSE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
SPARSE_MODEL_ID = "prithivida/Splade_PP_en_v1"  # SPLADE sparse
COLBERT_MODEL_ID = "colbert-ir/colbertv2.0"  # 128-dim multivector

dense_model = TextEmbedding(DENSE_MODEL_ID)
sparse_model = SparseTextEmbedding(SPARSE_MODEL_ID)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL_ID)

# Generate embeddings for all items
dense_embeds = list(
    dense_model.embed(texts, parallel=0)
)  # list[np.ndarray] shape (384,)

sparse_embeds = list(
    sparse_model.embed(texts, parallel=0)
)  # list[SparseEmbedding] with .indices/.values

colbert_multivectors = list(
    colbert_model.embed(texts, parallel=0)
)  # list[np.ndarray] shape (tokens, 128)


# Generate vectors for each item
points = []
for i, item in enumerate(sample_data):
    # Create sparse vector (keyword matching)
    sparse_vector = sparse_embeds[i].as_object()

    # Create dense vector (semantic understanding)
    dense_vector = dense_embeds[i]

    # Create ColBERT multivector (token-level understanding)
    colbert_vector = colbert_multivectors[i]

    points.append(
        models.PointStruct(
            id=i,
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector,
                "colbert": colbert_vector,
            },
            payload=item,
        )
    )

client.upload_points(collection_name=collection_name, points=points)
print(f"Uploaded {len(points)} recommendation items")



from datetime import timedelta

# Example user intent
user_query = "premium user likes sci-fi action movies with strong hacker themes"

user_dense_vector = next(dense_model.query_embed(user_query))
user_sparse_vector = next(sparse_model.query_embed(user_query)).as_object()
user_multivector = next(colbert_model.query_embed(user_query))


# Global filter - this will be propagated to ALL prefetch stages
global_filter = models.Filter(
    must=[
        # Content type and user segment
        models.FieldCondition(
            key="category", match=models.MatchValue(value="movie")
        ),
        models.FieldCondition(
            key="user_segment", match=models.MatchValue(value="premium")
        ),
        # Quality and recency constraints
        models.FieldCondition(
            key="release_date",
            range=models.DatetimeRange(
                gte=(datetime.now() - timedelta(days=365 * 30)).isoformat()
            ),
        ),
        models.FieldCondition(key="popularity_score", range=models.Range(gte=0.7)),
    ]
)

# Prefetch queries - global filter will be automatically applied to both
hybrid_query = [
    models.Prefetch(query=user_dense_vector, using="dense", limit=100),
    models.Prefetch(query=user_sparse_vector, using="sparse", limit=100),
]

# Fusion stage - combine candidates with RRF
fusion_query = models.Prefetch(
    prefetch=hybrid_query,
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=100,
)


# The Universal Query: Global filter propagates through all stages
response = client.query_points(
    collection_name=collection_name,
    prefetch=fusion_query,
    query=user_multivector,
    using="colbert",
    query_filter=global_filter,  # Propagates to all prefetch stages
    limit=10,
    with_payload=True,
)

for hit in response.points or []:
    print(hit.payload)


def build_recommendation_filter(user_profile, user_preference=None):
    """
    Build a global filter from user profile and preferences.
    This filter will automatically propagate to all prefetch stages.

    Args:
        user_profile: {
            "liked_titles": list[str],      # optional
            "preferred_genres": list[str],  # e.g. ["sci-fi","action"]
            "segment": str,                 # e.g. "premium"
            "query": str                    # free-text intent, e.g. "smart sci-fi with hacker vibe"
        }
        user_preference: {
            "category": str | None,         # e.g. "movie"
            "min_rating": float | None,     # e.g. 8.0
            "released_within_days": int | None  # e.g. 365
        }

    Returns:
        models.Filter object or None if no conditions
    """
    from datetime import datetime, timedelta

    filter_conditions = []

    # User segment filtering
    if user_profile.get("segment"):
        filter_conditions.append(
            models.FieldCondition(
                key="user_segment",
                match=models.MatchValue(value=user_profile["segment"]),
            )
        )

    if user_preference:
        # Category filtering
        if user_preference.get("category"):
            filter_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=user_preference["category"]),
                )
            )

        # Rating filtering
        if user_preference.get("min_rating") is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="rating",
                    range=models.Range(gte=user_preference["min_rating"])
                )
            )

        # Recency filtering
        if user_preference.get("released_within_days"):
            days = int(user_preference["released_within_days"])
            filter_conditions.append(
                models.FieldCondition(
                    key="release_date",
                    range=models.DatetimeRange(
                        gte=(datetime.utcnow() - timedelta(days=days)).isoformat()
                    ),
                )
            )

    return models.Filter(must=filter_conditions) if filter_conditions else None


def get_recommendations(user_profile, user_preference=None, limit=10):
    """
    Get personalized recommendations using Universal Query API

    Args:
        user_profile: {
            "liked_titles": list[str],      # optional
            "preferred_genres": list[str],  # e.g. ["sci-fi","action"]
            "segment": str,                 # e.g. "premium"
            "query": str                    # free-text intent, e.g. "smart sci-fi with hacker vibe"
        }
        user_preference: {
            "category": str | None,         # e.g. "movie"
            "min_rating": float | None,     # e.g. 8.0
            "released_within_days": int | None  # e.g. 365
        }
        limit: top-k to return
    """

    # Generate query embeddings
    user_dense_vector = next(dense_model.query_embed(user_profile["query"]))
    user_sparse_vector = next(
        sparse_model.query_embed(user_profile["query"])
    ).as_object()
    user_multivector = next(colbert_model.query_embed(user_profile["query"]))

    # Build global filter using helper function
    global_filter = build_recommendation_filter(user_profile, user_preference)

    # Prefetch queries - global filter will propagate automatically
    hybrid_query = [
        models.Prefetch(query=user_dense_vector, using="dense", limit=100),
        models.Prefetch(query=user_sparse_vector, using="sparse", limit=100),
    ]

    # Combine candidates with RRF
    fusion_query = models.Prefetch(
        prefetch=hybrid_query,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=100,
    )

    # Universal query - global filter propagates to all stages
    response = client.query_points(
        collection_name=collection_name,
        prefetch=fusion_query,
        query=user_multivector,
        using="colbert",
        query_filter=global_filter,  # Propagates to all prefetch stages
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "title": hit.payload["title"],
            "description": hit.payload["description"],
            "score": hit.score,
            "metadata": {
                k: v
                for k, v in hit.payload.items()
                if k not in ["title", "description"]
            },
        }
        for hit in (response.points or [])
    ]


# Test the recommendation service
user_profile = {
    "liked_titles": ["The Matrix", "Blade Runner"],
    "preferred_genres": ["sci-fi", "action"],
    "segment": "premium",
    "query": "highly rated cyberpunk movies",
}

recommendations = get_recommendations(
    user_profile,
    user_preference={
        "category": "movie",
        "min_rating": 8.0,
        "released_within_days": 365 * 30,
    },
    limit=10,
)

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']} (Score: {rec['score']:.3f})")


# Test DBSF vs RRF
# Filters and other prefetch params same as above

fusion_query = models.Prefetch(
    prefetch=hybrid_query,
    query=models.FusionQuery(fusion=models.Fusion.DBSF),
    limit=100,
)

response = client.query_points(
    collection_name=collection_name,
    prefetch=fusion_query,
    query=user_multivector,
    using="colbert",
    query_filter=global_filter,  # Same global filter propagates to all stages
    limit=10,
    with_payload=True,
)

for hit in response.points or []:
    print(hit.payload)
