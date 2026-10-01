"""
Lightweight dataclasses mirroring the PostgreSQL schema.

These are plain data holders (not an ORM). Repositories map raw
`dict_row` results from psycopg into these for type-safety and editor
support, without pulling in a heavyweight ORM layer.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class User:
    id: str
    email: str
    password_hash: str
    created_at: datetime
    updated_at: datetime


@dataclass
class Chat:
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False


@dataclass
class Document:
    id: str
    user_id: str
    chat_id: str
    filename: str
    file_type: str
    storage_reference: str
    status: str
    page_count: Optional[int]
    created_at: datetime


@dataclass
class Message:
    id: str
    chat_id: str
    user_id: str
    role: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class DocumentChunk:
    id: str
    user_id: str
    chat_id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]
