"""Relational persistence (SQLite by default, PostgreSQL in production via DATABASE_URL)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from geoai.schemas import AnalysisResult, AssetInfo, ReviewDecision, utcnow


class Base(DeclarativeBase):
    pass


class AssetRow(Base):
    __tablename__ = "assets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str | None] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(255))
    info_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisRow(Base):
    """Append-only provenance record. ``record_sha256`` makes tampering detectable."""

    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    domain: Mapped[str] = mapped_column(String(32))
    question: Mapped[str] = mapped_column(Text)
    needs_review: Mapped[int] = mapped_column(Integer, default=0, index=True)
    review_status: Mapped[str] = mapped_column(String(16), default="not_required")
    confidence: Mapped[float | None] = mapped_column(Float)
    record_sha256: Mapped[str | None] = mapped_column(String(64))
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ReviewRow(Base):
    """Human review decisions are stored separately and never mutate the original record."""

    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant: Mapped[str] = mapped_column(String(64), index=True)
    decision: Mapped[str] = mapped_column(String(16))
    reviewer: Mapped[str] = mapped_column(String(80))
    comment: Mapped[str | None] = mapped_column(Text)
    corrections_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Database:
    def __init__(self, url: str):
        kwargs: dict = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:  # a single shared connection keeps the in-memory DB alive across threads
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi, _):  # noqa: ANN001
                cur = dbapi.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        self._sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._sessions()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def ping(self) -> bool:
        with self.session() as s:
            s.execute(select(1))
        return True

    # ---- assets ----------------------------------------------------------
    def add_asset(self, info: AssetInfo, tenant: str, storage_key: str) -> None:
        with self.session() as s:
            s.add(AssetRow(id=info.id, tenant=tenant, filename=info.filename, sha256=info.sha256,
                           storage_key=storage_key, info_json=info.model_dump_json()))

    def get_asset(self, asset_id: str, tenant: str) -> tuple[AssetInfo, str] | None:
        with self.session() as s:
            row = s.get(AssetRow, asset_id)
            if row is None or row.tenant != tenant:  # tenant isolation: indistinguishable from "missing"
                return None
            return AssetInfo.model_validate_json(row.info_json), row.storage_key

    # ---- analyses --------------------------------------------------------
    def save_analysis(self, result: AnalysisResult, tenant: str) -> None:
        with self.session() as s:
            row = s.get(AnalysisRow, result.id)
            payload = result.model_dump_json()
            fields = dict(
                tenant=tenant, status=result.status, domain=result.domain.value, question=result.question,
                needs_review=int(result.needs_review), review_status=result.review_status,
                confidence=result.confidence,
                record_sha256=result.provenance.record_sha256 if result.provenance else None,
                result_json=payload, created_at=result.created_at,
            )
            if row is None:
                s.add(AnalysisRow(id=result.id, **fields))
            else:
                if row.status == "completed":
                    raise ValueError("completed analyses are immutable")
                for k, v in fields.items():
                    setattr(row, k, v)

    def get_analysis(self, analysis_id: str, tenant: str) -> AnalysisResult | None:
        with self.session() as s:
            row = s.get(AnalysisRow, analysis_id)
            if row is None or row.tenant != tenant:
                return None
            res = AnalysisResult.model_validate_json(row.result_json)
            res.review_status = row.review_status  # type: ignore[assignment]
            return res

    def list_analyses(self, tenant: str, limit: int = 50, needs_review: bool | None = None) -> list[AnalysisResult]:
        with self.session() as s:
            q = select(AnalysisRow).where(AnalysisRow.tenant == tenant).order_by(AnalysisRow.created_at.desc()).limit(limit)
            if needs_review is not None:
                q = q.where(AnalysisRow.needs_review == int(needs_review))
            out = []
            for row in s.scalars(q):
                res = AnalysisResult.model_validate_json(row.result_json)
                res.review_status = row.review_status  # type: ignore[assignment]
                out.append(res)
            return out

    def add_review(self, analysis_id: str, tenant: str, decision: ReviewDecision) -> bool:
        with self.session() as s:
            row = s.get(AnalysisRow, analysis_id)
            if row is None or row.tenant != tenant:
                return False
            s.add(ReviewRow(analysis_id=analysis_id, tenant=tenant, decision=decision.decision,
                            reviewer=decision.reviewer, comment=decision.comment,
                            corrections_json=json.dumps(decision.corrections)))
            row.review_status = decision.decision
            return True

    def list_reviews(self, analysis_id: str, tenant: str) -> list[dict]:
        with self.session() as s:
            rows = s.scalars(select(ReviewRow).where(ReviewRow.analysis_id == analysis_id, ReviewRow.tenant == tenant)
                             .order_by(ReviewRow.id))
            return [{"decision": r.decision, "reviewer": r.reviewer, "comment": r.comment,
                     "corrections": json.loads(r.corrections_json), "created_at": r.created_at.isoformat()} for r in rows]
