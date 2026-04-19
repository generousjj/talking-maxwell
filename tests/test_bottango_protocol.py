"""Bottango wire-protocol unit tests."""

from __future__ import annotations

from transport.bottango_protocol import (
    cmd_deregister_effector,
    cmd_handshake_request,
    cmd_instant_curve,
    cmd_register_pin_servo,
    compute_hash,
    frame_command,
    is_ok,
    normalized_to_compressed,
    parse_handshake_response,
)


def test_compute_hash_matches_documented_example():
    assert compute_hash("xUC,4") == 368


def test_frame_command_shape():
    assert frame_command("xUC,4") == b"xUC,4,h368\n"


def test_handshake_request_round_trip():
    payload = cmd_handshake_request(144)
    text = payload.decode("ascii")
    assert text.startswith("hRQ,144,h")
    assert text.endswith("\n")
    body = text.split(",h")[0]
    hash_value = int(text.rstrip("\n").split(",h")[-1])
    assert hash_value == sum(ord(c) for c in body)


def test_register_pin_servo_roundtrip():
    payload = cmd_register_pin_servo(
        pin=9, min_pwm=1450, max_pwm=1700, max_pwm_per_sec=3000, starting_pwm=1575
    )
    text = payload.decode("ascii")
    assert text.startswith("rSVPin,9,1450,1700,3000,1575,h")
    body, _, tail = text.rstrip("\n").partition(",h")
    assert int(tail) == sum(ord(c) for c in body)


def test_instant_curve_clamps_and_rounds():
    payload = cmd_instant_curve(9, 4096)
    text = payload.decode("ascii")
    assert text.startswith("sCI,9,4096,h")

    negative = cmd_instant_curve(9, -100)
    assert negative.decode("ascii").startswith("sCI,9,0,h")


def test_normalized_to_compressed_bounds():
    assert normalized_to_compressed(0.0) == 0
    assert normalized_to_compressed(1.0) == 8192
    assert normalized_to_compressed(-0.5) == 0
    assert normalized_to_compressed(1.5) == 8192
    assert 0 < normalized_to_compressed(0.5) <= 8192


def test_parse_handshake_response():
    parsed = parse_handshake_response("btngoHSK,0.7.0a1,12345,1")
    assert parsed == {"version": "0.7.0a1", "random_code": "12345", "accepting": True}
    assert parse_handshake_response("garbage") is None


def test_ok_detection():
    assert is_ok("OK")
    assert is_ok("  OK  ")
    assert not is_ok("BOOT")


def test_deregister_effector_command_hash_is_correct():
    payload = cmd_deregister_effector(9)
    text = payload.decode("ascii")
    body, _, tail = text.rstrip("\n").partition(",h")
    assert int(tail) == sum(ord(c) for c in body)
