from __future__ import annotations

from auth.auth_service import AuthService
from auth.password import hash_password, validate_password_strength, verify_password


def test_hash_and_verify_password_roundtrip():
    h = hash_password("Sup3rSecret")
    assert h != "Sup3rSecret"
    assert verify_password("Sup3rSecret", h)
    assert not verify_password("wrong-password", h)


def test_password_strength_rules():
    assert not validate_password_strength("short1A").ok  # too short
    assert not validate_password_strength("alllowercase1").ok  # no uppercase
    assert not validate_password_strength("ALLUPPERCASE1").ok  # no lowercase
    assert not validate_password_strength("NoDigitsHere").ok  # no digit
    assert validate_password_strength("GoodPass123").ok


def test_signup_success():
    service = AuthService()
    result = service.sign_up("new_user@example.com", "GoodPass123", "GoodPass123")
    assert result.ok
    assert result.user.email == "new_user@example.com"
    assert result.user.password_hash != "GoodPass123"


def test_signup_rejects_mismatched_confirmation():
    service = AuthService()
    result = service.sign_up("user2@example.com", "GoodPass123", "Different123")
    assert not result.ok
    assert any("match" in e.lower() for e in result.errors)


def test_signup_rejects_invalid_email():
    service = AuthService()
    result = service.sign_up("not-an-email", "GoodPass123", "GoodPass123")
    assert not result.ok


def test_signup_rejects_duplicate_email():
    service = AuthService()
    first = service.sign_up("dupe@example.com", "GoodPass123", "GoodPass123")
    assert first.ok

    second = service.sign_up("dupe@example.com", "AnotherPass123", "AnotherPass123")
    assert not second.ok
    assert "already exists" in second.error.lower()


def test_signup_email_is_case_insensitive_for_duplicates():
    service = AuthService()
    service.sign_up("CaseTest@Example.com", "GoodPass123", "GoodPass123")
    second = service.sign_up("casetest@example.com", "GoodPass123", "GoodPass123")
    assert not second.ok


def test_login_success():
    service = AuthService()
    service.sign_up("login_user@example.com", "GoodPass123", "GoodPass123")

    result = service.log_in("login_user@example.com", "GoodPass123")
    assert result.ok
    assert result.user.email == "login_user@example.com"


def test_login_rejects_incorrect_password():
    service = AuthService()
    service.sign_up("login_user2@example.com", "GoodPass123", "GoodPass123")

    result = service.log_in("login_user2@example.com", "WrongPassword1")
    assert not result.ok
    assert "invalid" in result.error.lower()


def test_login_rejects_unknown_email():
    service = AuthService()
    result = service.log_in("nobody@example.com", "GoodPass123")
    assert not result.ok
    assert "invalid" in result.error.lower()


def test_login_error_message_does_not_reveal_which_field_was_wrong():
    service = AuthService()
    service.sign_up("same_msg@example.com", "GoodPass123", "GoodPass123")

    wrong_password = service.log_in("same_msg@example.com", "WrongPassword1")
    unknown_email = service.log_in("nobody2@example.com", "WrongPassword1")
    assert wrong_password.error == unknown_email.error
