"""ActionCard schema (SRS 4.4). The single closed output object of the main agent.

`action` is always set by the deterministic policy, never the LLM (FR-34). `sigma` is
always present (rule 7): a model that returns None substitutes 0.5 and logs a warning
before this schema ever sees it, and the validator below still enforces non-null as a
second line of defense.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionEnum = Literal[
    "monitor", "prepare", "protect_asset", "evacuate_site", "inspect_after", "no_action"
]

FlagEnum = Literal[
    "perimeter_unofficial",
    "no_ros_high_sigma",
    "fhsz_missing",
    "firms_only",
    "stale_E",
    "aoi_coarse_advisory",
    "firms_unavailable",
    "degraded",
]


class SiteRef(BaseModel):
    site_id: str
    name: str
    lat: float
    lng: float
    mode: Literal["point", "parcel", "aoi"] = "point"
    geocode_confidence: str = "n/a"
    range_interpolation: bool = False


class IncidentInfo(BaseModel):
    irwin_id: Optional[str] = None
    incident_name: Optional[str] = None
    acres: Optional[float] = None
    containment_pct: Optional[float] = None
    dist_perimeter_m: Optional[float] = None
    hours_since_discovery: Optional[float] = None


class WeatherInfo(BaseModel):
    red_flag: bool = False
    wind_speed_ms: Optional[float] = None
    wind_dir_cardinal: Optional[str] = None
    rh_pct: Optional[float] = None
    hrrr_valid_time: Optional[str] = None


class PBurnByT(BaseModel):
    t24: Optional[float] = Field(default=None, alias="24")
    t48: Optional[float] = Field(default=None, alias="48")
    t72: Optional[float] = Field(default=None, alias="72")

    model_config = ConfigDict(populate_by_name=True)


class Citation(BaseModel):
    source: str
    url: str
    fetched_at: str
    field: Optional[str] = None


class WSource(BaseModel):
    field: str
    source_url: str
    vintage: Optional[str] = None
    confidence: Optional[str] = None


class EProductTime(BaseModel):
    feed: str
    product_time: str


class ActionCard(BaseModel):
    card_id: str
    generated_at: str
    site: SiteRef
    action: ActionEnum
    eta_hours: Optional[float] = None
    eta_sigma_hours: Optional[float] = None
    p_burn_by_T: Optional[PBurnByT] = None
    y_hat: Optional[float] = None
    sigma: float
    baseline_y: Optional[float] = None
    incident: IncidentInfo
    weather: WeatherInfo
    recommended_actions: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    flags: list[FlagEnum] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    e_product_times: list[EProductTime] = Field(default_factory=list)
    w_sources: list[WSource] = Field(default_factory=list)
    model_version: str
    spread_field_version: Optional[str] = None
    policy_version: str

    @model_validator(mode="after")
    def sigma_always_present(self) -> "ActionCard":
        if self.sigma is None:
            raise ValueError("sigma must always be set (SRS rule 7 / NFR-17)")
        return self
