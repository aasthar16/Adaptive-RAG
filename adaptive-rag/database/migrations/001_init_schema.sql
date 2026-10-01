-- AdaptiveRAG initial schema
-- Safe to re-run: every statement is idempotent.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- users
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

-- ---------------------------------------------------------------------
-- chats
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chats (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    title           TEXT NOT NULL DEFAULT 'New Chat',
    is_deleted      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats (user_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_chats_user_updated ON chats (user_id, updated_at DESC);

-- ---------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    chat_id             UUID NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    filename            TEXT NOT NULL,
    file_type           TEXT NOT NULL,
    storage_reference   TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending', -- pending|processing|indexed|failed
    page_count          INTEGER,
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_chat_id ON documents (chat_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id);

-- ---------------------------------------------------------------------
-- messages
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id         UUID NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    sources         JSONB NOT NULL DEFAULT '[]'::jsonb,
    debug_info      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages (chat_id, created_at);

-- ---------------------------------------------------------------------
-- document_chunks (vector store, queried directly by langchain-postgres
-- PGVectorStore AND by our own keyword/hybrid retrievers)
-- ---------------------------------------------------------------------
-- NOTE: embedding dimensionality must match config.settings.EMBEDDING_DIMENSIONS.
-- Default: text-embedding-3-small -> 1536 dims.
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    chat_id         UUID NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    embedding       vector(1536),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_chat_id ON document_chunks (chat_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN (content_tsv);

-- Approximate nearest-neighbour index for cosine similarity search.
-- HNSW gives better recall/latency than IVFFlat and needs no training step,
-- so it works even on a freshly created, empty table.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- updated_at triggers
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_chats_updated_at ON chats;
CREATE TRIGGER trg_chats_updated_at
    BEFORE UPDATE ON chats
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
