"""SQLAlchemy ORM models for the FSLSM-RAG-MCP thesis project."""
from __future__ import annotations

import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ============================================================
# CORPUS & RETRIEVAL
# ============================================================

class Textbook(Base):
    __tablename__ = "textbooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[Optional[str]] = mapped_column(String(50))
    source_path: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="textbook")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    textbook_id: Mapped[Optional[int]] = mapped_column(ForeignKey("textbooks.id"))
    chapter: Mapped[Optional[str]] = mapped_column(String(100))
    section: Mapped[Optional[str]] = mapped_column(String(200))
    concept_labels: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))  # MiniLM dim
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    textbook: Mapped[Optional["Textbook"]] = relationship("Textbook", back_populates="chunks")


# ============================================================
# FSLSM PROFILES
# ============================================================

class FslsmProfile(Base):
    __tablename__ = "fslsm_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    # Binary dimensions: -1 or +1 per pole
    act_ref: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sen_int: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    vis_ver: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    seq_glo: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text)
    style_descriptor: Mapped[Optional[str]] = mapped_column(Text)
    behavioral_instructions: Mapped[Optional[dict]] = mapped_column(JSONB)

    agents: Mapped[list["Agent"]] = relationship("Agent", back_populates="profile")

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "act_ref": self.act_ref,
            "sen_int": self.sen_int,
            "vis_ver": self.vis_ver,
            "seq_glo": self.seq_glo,
        }


# ============================================================
# AGENTS (EXPERIMENT 1+)
# ============================================================

class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_uid: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    profile_id: Mapped[Optional[int]] = mapped_column(ForeignKey("fslsm_profiles.id"))
    instance_num: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    llm_model: Mapped[Optional[str]] = mapped_column(String(100))
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    profile: Mapped[Optional["FslsmProfile"]] = relationship(
        "FslsmProfile", back_populates="agents"
    )
    logs: Mapped[list["InteractionLog"]] = relationship(
        "InteractionLog", back_populates="agent"
    )


# ============================================================
# GROUND TRUTH (MULTI-HOP QA)
# ============================================================

class EvalQuestion(Base):
    __tablename__ = "eval_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    gold_answer: Mapped[str] = mapped_column(Text, nullable=False)
    answer_type: Mapped[Optional[str]] = mapped_column(String(50))
    difficulty: Mapped[Optional[str]] = mapped_column(String(20))
    quality_tier: Mapped[Optional[str]] = mapped_column(String(10))
    gold_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    essential_ids: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))
    supporting_ids: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))

    logs: Mapped[list["InteractionLog"]] = relationship(
        "InteractionLog", back_populates="question"
    )


# ============================================================
# EXPERIMENT RUNS & INTERACTION LOGS
# ============================================================

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment: Mapped[str] = mapped_column(String(10), nullable=False)  # "exp1", "exp2", "exp3"
    config: Mapped[Optional[dict]] = mapped_column(JSONB)
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))

    logs: Mapped[list["InteractionLog"]] = relationship(
        "InteractionLog", back_populates="run"
    )


class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("experiment_runs.id"))
    agent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agents.id"))
    question_id: Mapped[Optional[int]] = mapped_column(ForeignKey("eval_questions.id"))
    system_variant: Mapped[Optional[str]] = mapped_column(String(10))  # R0/R1, S0/S1a/S1b

    # Request / Response
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)

    # Retrieved context
    retrieved_chunk_ids: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))
    selected_tool: Mapped[Optional[str]] = mapped_column(String(100))  # exp3

    # Raw outputs
    tutor_response: Mapped[Optional[str]] = mapped_column(Text)
    agent_response: Mapped[Optional[str]] = mapped_column(Text)

    # Computed metrics (nullable — filled by evaluate step)
    pra_score: Mapped[Optional[float]] = mapped_column(Float)       # exp1
    das_scores: Mapped[Optional[dict]] = mapped_column(JSONB)        # exp1 per-dim
    scs_score: Mapped[Optional[float]] = mapped_column(Float)        # exp2
    rr_score: Mapped[Optional[float]] = mapped_column(Float)         # exp2
    engagement: Mapped[Optional[float]] = mapped_column(Float)       # exp2
    chunk_recall_5: Mapped[Optional[float]] = mapped_column(Float)   # exp2
    tsa_correct: Mapped[Optional[bool]] = mapped_column(Boolean)     # exp3

    # Extra JSON blob for anything not covered above
    extra: Mapped[Optional[dict]] = mapped_column(JSONB)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run: Mapped[Optional["ExperimentRun"]] = relationship(
        "ExperimentRun", back_populates="logs"
    )
    agent: Mapped[Optional["Agent"]] = relationship("Agent", back_populates="logs")
    question: Mapped[Optional["EvalQuestion"]] = relationship(
        "EvalQuestion", back_populates="logs"
    )
