"""
Password hashing (Argon2id) and basic strength validation.

Argon2 is used over bcrypt because it's the current OWASP-recommended
default for new applications and has no 72-byte input truncation footgun.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError, VerificationError

_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,  # 64 MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

MIN_PASSWORD_LENGTH = 8


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


def hash_password(plain_password: str) -> str:
    return _hasher.hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, plain_password)
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True if the hash was created with weaker parameters than current."""
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip())) and len(email) <= 254


def validate_password_strength(password: str) -> ValidationResult:
    errors: list[str] = []
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter.")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one digit.")
    return ValidationResult(ok=not errors, errors=errors)


def validate_signup(email: str, password: str, confirm_password: str) -> ValidationResult:
    errors: list[str] = []
    if not is_valid_email(email):
        errors.append("Please enter a valid email address.")
    strength = validate_password_strength(password)
    errors.extend(strength.errors)
    if password != confirm_password:
        errors.append("Passwords do not match.")
    return ValidationResult(ok=not errors, errors=errors)
