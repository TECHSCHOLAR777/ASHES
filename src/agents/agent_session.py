"""Mutable working memory for one agentic ask. Tools read/write this; the LLM does not."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.agents.main_agent import MainAgentDeps, Site
from src.clients.firms import FIRMSResult
from src.clients.hrrr import HRRRWeather
from src.clients.nws import CAPAlert, SPCOutlook
from src.clients.wfigs import WFIGSIncident, WFIGSPerimeter
from src.features.e_packer import EFeatures
from src.features.w_encoder import WFeatures
from src.geometry.aoi import Geometry
from src.model.h_fire import ModelOutput
from src.policy.engine import PolicyResult
from src.schemas.action_card import ActionCard
from src.spread.client import SpreadField, SpreadSiteSample


EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    backfill: bool = False


@dataclass
class AgentSession:
    deps: MainAgentDeps
    site: Site
    question: str
    simulate: bool = False
    ignition_lat: float | None = None
    ignition_lng: float | None = None
    buffer_km: float | None = None
    on_event: EventCallback | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    geocode_confidence: str = "n/a"
    range_interpolation: bool = False
    parse: dict[str, Any] | None = None
    playbook_steps: list[str] = field(default_factory=list)
    notify: dict[str, Any] | None = None
    coords_supplied: bool = True
    situation_injected: bool = False
    cap_alerts: list[CAPAlert] = field(default_factory=list)
    spc: Optional[SPCOutlook] = None
    firms: Optional[FIRMSResult] = None
    incidents: list[WFIGSIncident] = field(default_factory=list)
    perimeters: list[WFIGSPerimeter] = field(default_factory=list)
    hrrr: Optional[HRRRWeather] = None
    raw_w: dict[str, Any] = field(default_factory=dict)
    quoted_credits: float | None = None
    flags: list[str] = field(default_factory=list)
    e_features: Optional[EFeatures] = None
    w_features: Optional[WFeatures] = None
    spread_field: Optional[SpreadField] = None
    spread_sample: Optional[SpreadSiteSample] = None
    geom: Optional[Geometry] = None
    model_output: Optional[ModelOutput] = None
    policy_result: Optional[PolicyResult] = None
    action_card: Optional[ActionCard] = None
    response_card: Any = None
    brief: Optional[str] = None
    brief_replaced: bool = False
    spread_viz: dict[str, Any] | None = None
    traces: list[ToolTrace] = field(default_factory=list)
    called: set[str] = field(default_factory=set)
    committed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)
