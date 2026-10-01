"""
Section 38: "chat survives application restart", "messages survive
restart", "LangGraph state survives restart".

Streamlit reruns the whole script on every interaction and the process can
restart entirely between sessions, so nothing may live only in memory.
These tests simulate a restart by tearing down the connection pool (and,
for the LangGraph case, creating a brand new checkpointer/graph instance)
and confirming everything is still readable afterwards, exactly as a fresh
process would see it.
"""
from __future__ import annotations

from database.connection import close_pool
from database.repositories.chat_repository import ChatRepository
from database.repositories.message_repository import MessageRepository


def _simulate_process_restart():
    """Close the pooled connections; the next repository call will lazily
    open a brand new pool, just like a fresh `streamlit run app.py`."""
    close_pool()


def test_chat_survives_simulated_restart(make_user):
    user = make_user()
    chats = ChatRepository()
    chat = chats.create(user_id=user.id, title="Survives restart")

    _simulate_process_restart()

    reloaded = ChatRepository().get_owned_by_user(chat.id, user.id)
    assert reloaded is not None
    assert reloaded.title == "Survives restart"


def test_messages_survive_simulated_restart(make_user):
    user = make_user()
    chats = ChatRepository()
    messages = MessageRepository()
    chat = chats.create(user_id=user.id, title="Chat with history")

    messages.create(chat.id, user.id, role="user", content="What is gradient descent?")
    messages.create(
        chat.id,
        user.id,
        role="assistant",
        content="Gradient descent is an optimization algorithm. [1]",
        sources=[{"index": 1, "filename": "ml.pdf", "page_number": 3, "chunk_id": "abc"}],
        debug_info={"iterations": 1},
    )

    _simulate_process_restart()

    reloaded = MessageRepository().list_for_chat(chat.id)
    assert len(reloaded) == 2
    assert reloaded[0].role == "user"
    assert reloaded[1].role == "assistant"
    assert reloaded[1].sources[0]["filename"] == "ml.pdf"
    assert reloaded[1].debug_info["iterations"] == 1


def test_langgraph_thread_state_survives_simulated_restart(make_user):
    """A minimal check that a LangGraph checkpoint written under a given
    thread_id is still readable via a brand-new PostgresSaver + compiled
    graph — i.e. conversational graph state is not tied to any in-memory
    Streamlit session."""
    from langgraph.graph import END, START, StateGraph
    from typing import TypedDict

    from config.settings import get_settings
    from langgraph.checkpoint.postgres import PostgresSaver

    class TinyState(TypedDict):
        counter: int

    def bump(state: TinyState) -> dict:
        return {"counter": state["counter"] + 1}

    def _build():
        g = StateGraph(TinyState)
        g.add_node("bump", bump)
        g.add_edge(START, "bump")
        g.add_edge("bump", END)
        return g

    settings = get_settings()
    thread_id = "persistence-test-thread"
    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(settings.database_url_psycopg) as saver:
        graph = _build().compile(checkpointer=saver)
        graph.invoke({"counter": 0}, config=config)

    _simulate_process_restart()

    # Brand new saver/graph instance, simulating a fresh process — the
    # checkpoint history must still be there under the same thread_id.
    with PostgresSaver.from_conn_string(get_settings().database_url_psycopg) as saver:
        graph = _build().compile(checkpointer=saver)
        state = graph.get_state(config)
        assert state is not None
        assert state.values["counter"] == 1
