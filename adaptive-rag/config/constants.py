"""Static constants that are not environment-dependent configuration."""
from enum import Enum


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    WEB = "web"


class QueryType(str, Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    EXACT_LOOKUP = "exact_lookup"
    SUMMARIZATION = "summarization"
    OUT_OF_SCOPE = "out_of_scope"


class QueryComplexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvaluationClassification(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    POOR = "poor"


class CorrectionAction(str, Enum):
    NONE = "none"
    RERANK = "rerank"
    REWRITE_QUERY = "rewrite_query"
    ADDITIONAL_RETRIEVAL = "additional_retrieval"
    EXTERNAL_SEARCH = "external_search"
    GIVE_UP = "give_up"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Roughly 4 characters per token; used only for cheap, approximate cost
# tracking in the observability panel (never billed against, just displayed).
CHARS_PER_TOKEN_ESTIMATE = 4

VECTOR_TABLE_NAME = "document_chunks"
CHECKPOINT_SCHEMA_INITIALIZED_FLAG = "_langgraph_checkpoint_initialized"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}
