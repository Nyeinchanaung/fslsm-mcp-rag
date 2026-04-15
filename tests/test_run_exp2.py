"""Tests for Experiment 2 A/B Runner (Phase 4).

Unit tests for session list building, checkpoint/resume, and CLI parsing.
"""

import json
import tempfile
from pathlib import Path

import pytest

from experiments.exp2_tutor_personalization.run_exp2 import (
    build_session_list,
    filter_remaining,
    load_completed_keys,
    parse_args,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

SAMPLE_AGENTS = [
    {
        "agent_uid": "A01",
        "profile_label": "Active-Sensing-Visual-Sequential",
        "fslsm_vector": {"act_ref": -1, "sen_int": -1, "vis_ver": -1, "seq_glo": -1},
    },
    {
        "agent_uid": "A02",
        "profile_label": "Reflective-Intuitive-Verbal-Global",
        "fslsm_vector": {"act_ref": 1, "sen_int": 1, "vis_ver": 1, "seq_glo": 1},
    },
]

SAMPLE_QUESTIONS = [
    {"question_id": "Q1", "question": "What is gradient descent?"},
    {"question_id": "Q2", "question": "Explain backpropagation."},
]


# -------------------------------------------------------------------
# build_session_list
# -------------------------------------------------------------------

class TestBuildSessionList:
    def test_total_count(self):
        sessions = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0", "R1"])
        # 2 agents x 2 questions x 2 modes = 8
        assert len(sessions) == 8

    def test_single_mode(self):
        sessions = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0"])
        assert len(sessions) == 4
        assert all(s["mode"] == "R0" for s in sessions)

    def test_session_keys(self):
        sessions = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0"])
        required_keys = {"agent_id", "profile_label", "fslsm_vector", "question_id", "question", "mode"}
        for s in sessions:
            assert required_keys.issubset(s.keys())

    def test_deterministic_shuffle(self):
        s1 = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0", "R1"], seed=42)
        s2 = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0", "R1"], seed=42)
        assert [s["agent_id"] for s in s1] == [s["agent_id"] for s in s2]

    def test_different_seed_different_order(self):
        s1 = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0", "R1"], seed=42)
        s2 = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0", "R1"], seed=99)
        order1 = [(s["agent_id"], s["question_id"], s["mode"]) for s in s1]
        order2 = [(s["agent_id"], s["question_id"], s["mode"]) for s in s2]
        assert order1 != order2


# -------------------------------------------------------------------
# Checkpoint / Resume
# -------------------------------------------------------------------

class TestCheckpointResume:
    def test_load_completed_keys_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.touch()
        assert load_completed_keys(f) == set()

    def test_load_completed_keys_nonexistent(self, tmp_path):
        f = tmp_path / "missing.jsonl"
        assert load_completed_keys(f) == set()

    def test_load_completed_keys_with_data(self, tmp_path):
        f = tmp_path / "sessions.jsonl"
        records = [
            {"agent_id": "A01", "question_id": "Q1", "mode": "R0", "response": "..."},
            {"agent_id": "A02", "question_id": "Q2", "mode": "R1", "response": "..."},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        completed = load_completed_keys(f)
        assert len(completed) == 2
        assert ("A01", "Q1", "R0") in completed
        assert ("A02", "Q2", "R1") in completed

    def test_load_completed_keys_skips_bad_lines(self, tmp_path):
        f = tmp_path / "sessions.jsonl"
        lines = [
            json.dumps({"agent_id": "A01", "question_id": "Q1", "mode": "R0"}),
            "not valid json",
            "",
            json.dumps({"agent_id": "A02"}),  # missing keys
        ]
        f.write_text("\n".join(lines) + "\n")
        completed = load_completed_keys(f)
        assert len(completed) == 1

    def test_filter_remaining(self):
        sessions = build_session_list(SAMPLE_AGENTS, SAMPLE_QUESTIONS, ["R0"])
        completed = {("A01", "Q1", "R0")}
        remaining = filter_remaining(sessions, completed)
        assert len(remaining) == 3
        assert not any(
            s["agent_id"] == "A01" and s["question_id"] == "Q1"
            for s in remaining
        )


# -------------------------------------------------------------------
# CLI argument parsing
# -------------------------------------------------------------------

class TestCLIParsing:
    def test_defaults(self):
        args = parse_args([])
        assert args.mode == "both"
        assert args.workers == 5
        assert args.dry_run is False
        assert args.n is None

    def test_mode_r0(self):
        args = parse_args(["--mode", "R0"])
        assert args.mode == "R0"

    def test_workers(self):
        args = parse_args(["--workers", "10"])
        assert args.workers == 10

    def test_dry_run(self):
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_n_limit(self):
        args = parse_args(["--n", "4"])
        assert args.n == 4
