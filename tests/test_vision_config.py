"""VisionConfig loads from YAML onto AppConfig via the dataclass walker."""

from __future__ import annotations

import textwrap

import yaml

from app.config import AppConfig, VisionConfig, _apply


def test_vision_config_defaults():
    cfg = AppConfig()
    assert isinstance(cfg.vision, VisionConfig)
    assert cfg.vision.enabled is False
    assert cfg.vision.camera_index == 0
    assert cfg.vision.scene_model == "gpt-4o-mini"


def test_vision_block_overrides_defaults():
    raw = yaml.safe_load(
        textwrap.dedent(
            """
            vision:
              enabled: true
              camera_index: 2
              tracking_fps: 20
              gaze_gain_lr: 2.0
              invert_lr: true
              scene_model: gpt-4o
              scene_min_interval_s: 5.0
            """
        )
    )
    cfg = AppConfig()
    _apply(cfg, raw)
    assert cfg.vision.enabled is True
    assert cfg.vision.camera_index == 2
    assert cfg.vision.tracking_fps == 20
    assert cfg.vision.gaze_gain_lr == 2.0
    assert cfg.vision.invert_lr is True
    assert cfg.vision.scene_model == "gpt-4o"
    assert cfg.vision.scene_min_interval_s == 5.0
    # Untouched fields keep their defaults.
    assert cfg.vision.invert_ud is False
    assert cfg.vision.deadzone == 0.03


# ---- feature switches: which voice tools each vision setting exposes ----


def _pipeline_with(**vision):
    from types import SimpleNamespace

    from app.pipeline import ConversationPipeline
    from conversation.llm import StubLLM
    from motion.state_machine import ConversationStateMachine

    return ConversationPipeline(
        stt=None, llm=StubLLM(), tts=None, backend=None,
        state_machine=ConversationStateMachine(),
        jaw_calibration=None, behavior_gains=None,
        vision_config=SimpleNamespace(**vision),
    )


def _tool_names(p):
    tools, _hint = p._build_realtime_tools()
    return {t["name"] for t in tools}


def test_scene_and_memory_off_exposes_no_tools():
    # Friday booth config: tracking only — no paid look tool, no face memory.
    p = _pipeline_with(enabled=True, scene_enabled=False, recognition_enabled=False)
    assert _tool_names(p) == set()


def test_each_switch_adds_only_its_own_tools():
    assert _tool_names(_pipeline_with(scene_enabled=True, recognition_enabled=False)) == {
        "look_and_describe"
    }
    assert _tool_names(_pipeline_with(scene_enabled=False, recognition_enabled=True)) == {
        "remember_person",
        "forget_person",
    }


def test_describe_scene_never_calls_api_when_scene_disabled():
    import asyncio

    p = _pipeline_with(scene_enabled=False)
    reply = asyncio.run(p.describe_scene())
    assert "can't describe" in reply
