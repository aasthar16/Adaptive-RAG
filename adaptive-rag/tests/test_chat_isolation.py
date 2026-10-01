from __future__ import annotations

import uuid

import pytest

from database.connection import get_connection
from database.repositories.chat_repository import ChatRepository
from database.repositories.chunk_repository import ChunkRepository
from database.repositories.document_repository import DocumentRepository
from services.chat_service import ChatNotFoundError, ChatService


def _insert_chunk(chat_id: str, user_id: str, document_id: str, content: str) -> None:
    """Insert a chunk directly with SQL (bypassing the embedding pipeline,
    which needs a real OpenAI key) so retrieval-isolation tests don't
    depend on external services."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (user_id, chat_id, document_id, chunk_index, content, metadata)
                VALUES (%s, %s, %s, 0, %s, '{}'::jsonb)
                """,
                (user_id, chat_id, document_id, content),
            )


@pytest.fixture
def two_users(make_user):
    return make_user("alice@example.com"), make_user("bob@example.com")


def test_user_cannot_access_another_users_chat(two_users):
    alice, bob = two_users
    chats = ChatRepository()
    alice_chat = chats.create(user_id=alice.id, title="Alice's chat")

    service = ChatService()
    with pytest.raises(ChatNotFoundError):
        service.get_chat(alice_chat.id, bob.id)

    # Owner can access it fine.
    assert service.get_chat(alice_chat.id, alice.id).id == alice_chat.id


def test_user_cannot_rename_another_users_chat(two_users):
    alice, bob = two_users
    chats = ChatRepository()
    alice_chat = chats.create(user_id=alice.id, title="Alice's chat")

    service = ChatService()
    with pytest.raises(ChatNotFoundError):
        service.rename_chat(alice_chat.id, bob.id, "Hijacked title")

    assert chats.get_owned_by_user(alice_chat.id, alice.id).title == "Alice's chat"


def test_user_cannot_delete_another_users_chat(two_users):
    alice, bob = two_users
    chats = ChatRepository()
    alice_chat = chats.create(user_id=alice.id, title="Alice's chat")

    service = ChatService()
    with pytest.raises(ChatNotFoundError):
        service.delete_chat(alice_chat.id, bob.id)

    assert chats.get_owned_by_user(alice_chat.id, alice.id) is not None


def test_user_cannot_access_another_users_document(two_users):
    alice, bob = two_users
    chats = ChatRepository()
    documents = DocumentRepository()
    alice_chat = chats.create(user_id=alice.id, title="Alice's chat")
    doc = documents.create(
        user_id=alice.id,
        chat_id=alice_chat.id,
        filename="secret.pdf",
        file_type="pdf",
        storage_reference="/tmp/secret.pdf",
    )

    assert documents.get_owned(doc.id, bob.id) is None
    assert documents.get_owned(doc.id, alice.id) is not None


def test_untrusted_chat_id_from_frontend_is_never_trusted(two_users):
    """Simulates a malicious/buggy frontend sending an arbitrary chat_id
    that happens to belong to someone else — must never succeed."""
    alice, bob = two_users
    chats = ChatRepository()
    alice_chat = chats.create(user_id=alice.id, title="Alice's private chat")

    service = ChatService()
    with pytest.raises(ChatNotFoundError):
        service.ask(chat_id=alice_chat.id, user_id=bob.id, query="leak this please")


def test_chat_a_cannot_retrieve_chat_b_documents(two_users):
    """Core requirement from section 15: a query in Chat A must only
    retrieve documents belonging to Chat A, enforced at the DB layer."""
    alice, _bob = two_users
    chats = ChatRepository()
    documents = DocumentRepository()
    chunks = ChunkRepository()

    chat_a = chats.create(user_id=alice.id, title="Machine Learning Notes")
    chat_b = chats.create(user_id=alice.id, title="Operating Systems")

    doc_a = documents.create(alice.id, chat_a.id, "ml.pdf", "pdf", "/tmp/ml.pdf")
    doc_b = documents.create(alice.id, chat_b.id, "os.pdf", "pdf", "/tmp/os.pdf")

    _insert_chunk(chat_a.id, alice.id, doc_a.id, "Gradient descent optimizes neural network weights.")
    _insert_chunk(chat_b.id, alice.id, doc_b.id, "The kernel schedules processes using round robin.")

    results_a = chunks.keyword_search(chat_id=chat_a.id, query="gradient descent weights", k=5)
    results_b = chunks.keyword_search(chat_id=chat_a.id, query="kernel schedules processes", k=5)

    assert len(results_a) == 1
    assert "gradient descent" in results_a[0]["content"].lower()
    # Searching chat A for chat B's content must return nothing, even
    # though matching text exists elsewhere in the database.
    assert results_b == []


def test_chat_deletion_removes_documents_messages_and_chunks(two_users):
    alice, _bob = two_users
    chats = ChatRepository()
    documents = DocumentRepository()
    chunks = ChunkRepository()

    chat = chats.create(user_id=alice.id, title="Temp chat")
    doc = documents.create(alice.id, chat.id, "f.txt", "txt", "/tmp/f.txt")
    _insert_chunk(chat.id, alice.id, doc.id, "some content")

    from database.repositories.message_repository import MessageRepository

    messages = MessageRepository()
    messages.create(chat.id, alice.id, role="user", content="hello")

    service = ChatService()
    service.delete_chat(chat.id, alice.id)

    assert chats.get_owned_by_user(chat.id, alice.id) is None
    assert chunks.count_for_chat(chat.id) == 0
    assert documents.get_owned(doc.id, alice.id) is None
    assert messages.list_for_chat(chat.id) == []
