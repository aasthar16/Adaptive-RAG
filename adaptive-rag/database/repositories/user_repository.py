from __future__ import annotations

from typing import Optional

from database.connection import get_connection
from database.models import User


class UserRepository:
    def create(self, email: str, password_hash: str) -> User:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (email, password_hash)
                    VALUES (%s, %s)
                    RETURNING id, email, password_hash, created_at, updated_at
                    """,
                    (email.lower().strip(), password_hash),
                )
                row = cur.fetchone()
        return User(**row)

    def get_by_email(self, email: str) -> Optional[User]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, email, password_hash, created_at, updated_at
                    FROM users WHERE email = %s
                    """,
                    (email.lower().strip(),),
                )
                row = cur.fetchone()
        return User(**row) if row else None

    def get_by_id(self, user_id: str) -> Optional[User]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, email, password_hash, created_at, updated_at
                    FROM users WHERE id = %s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        return User(**row) if row else None

    def email_exists(self, email: str) -> bool:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE email = %s", (email.lower().strip(),))
                return cur.fetchone() is not None

    def update_password_hash(self, user_id: str, new_hash: str) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (new_hash, user_id),
                )
