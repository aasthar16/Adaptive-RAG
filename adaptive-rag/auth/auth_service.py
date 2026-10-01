from __future__ import annotations

from dataclasses import dataclass

from auth.password import hash_password, needs_rehash, validate_signup, verify_password
from database.models import User
from database.repositories.user_repository import UserRepository


@dataclass
class AuthResult:
    ok: bool
    user: User | None = None
    errors: list[str] | None = None

    @property
    def error(self) -> str:
        return "; ".join(self.errors or [])


class AuthService:
    """Signup / login orchestration. Never leaks whether an email exists
    beyond what is strictly necessary for a usable error message."""

    def __init__(self, user_repository: UserRepository | None = None) -> None:
        self.users = user_repository or UserRepository()

    def sign_up(self, email: str, password: str, confirm_password: str) -> AuthResult:
        email = email.strip().lower()
        validation = validate_signup(email, password, confirm_password)
        if not validation.ok:
            return AuthResult(ok=False, errors=validation.errors)

        if self.users.email_exists(email):
            return AuthResult(ok=False, errors=["An account with this email already exists."])

        password_hash = hash_password(password)
        try:
            user = self.users.create(email=email, password_hash=password_hash)
        except Exception as exc:  # e.g. unique-constraint race condition
            return AuthResult(ok=False, errors=[f"Could not create account: {exc}"])
        return AuthResult(ok=True, user=user)

    def log_in(self, email: str, password: str) -> AuthResult:
        email = email.strip().lower()
        user = self.users.get_by_email(email)
        # Always run verify_password even on a missing user, using a dummy
        # hash, so response timing doesn't reveal whether the email exists.
        dummy_hash = (
            "$argon2id$v=19$m=65536,t=3,p=2$"
            "MDEyMzQ1Njc4OWFiY2RlZg$0123456789abcdef0123456789abcdef"
        )
        candidate_hash = user.password_hash if user else dummy_hash
        password_ok = verify_password(password, candidate_hash)

        if not user or not password_ok:
            return AuthResult(ok=False, errors=["Invalid email or password."])

        if needs_rehash(user.password_hash):
            new_hash = hash_password(password)
            self.users.update_password_hash(user.id, new_hash)

        return AuthResult(ok=True, user=user)
