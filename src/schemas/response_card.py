"""ResponseCard schema (SRS 4.5). Emitted by the Response Support Agent, no trained model.

Every number here traces to a Mireye field, a live USGS gage, or deterministic code -
never the LLM (FR-55).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

WaterSourceType = Literal[
    "stream", "lake_reservoir", "wetland", "municipal_water", "well_field",
    "wastewater_plant", "dam_reservoir",
]
AvailabilityEnum = Literal["high", "moderate", "low", "dry", "unknown"]
UsabilityEnum = Literal["excellent", "good", "limited", "unknown"]
HazmatType = Literal[
    "EPA_RMP", "RCRA_TSD", "UST", "gas_pipeline", "petroleum_pipeline",
    "transmission_line", "osm_substation",
]
PriorityEnum = Literal["critical", "high", "medium"]
ConstraintType = Literal["critical_habitat", "protected_area"]
ConstraintEnum = Literal["retardant_restricted", "access_restricted", "coordinate_first", "normal"]
DischargeClassEnum = Literal["high", "normal", "low", "critically_low", "unknown"]


class SiteRef(BaseModel):
    site_id: str
    name: str
    lat: float
    lng: float


class WaterSource(BaseModel):
    type: WaterSourceType
    name: Optional[str] = None
    distance_m: float
    discharge_cfs: Optional[float] = None
    permanence_pct: Optional[float] = None
    availability: AvailabilityEnum
    note: Optional[str] = None
    source_url: str
    fetched_at: str


class AccessRoute(BaseModel):
    road_name: Optional[str] = None
    road_class: str
    surface: str
    distance_m: float
    usability: UsabilityEnum


class FireStation(BaseModel):
    name: Optional[str] = None
    distance_m: float
    eta_minutes_estimate: Optional[float] = None


class Airport(BaseModel):
    name: Optional[str] = None
    distance_m: float


class HazmatSite(BaseModel):
    type: HazmatType
    name: Optional[str] = None
    distance_m: float
    priority: PriorityEnum
    note: Optional[str] = None


class EnvironmentalConstraint(BaseModel):
    type: ConstraintType
    species_or_designation: Optional[str] = None
    manager: Optional[str] = None
    constraint: ConstraintEnum


class Comms(BaseModel):
    mobile_5g_coverage: Optional[str] = None
    nearest_antenna_distance_m: Optional[float] = None
    nearest_antenna_height_m: Optional[float] = None
    fiber_available: Optional[bool] = None


class HospitalRef(BaseModel):
    name: Optional[str] = None
    distance_m: Optional[float] = None


class Evacuation(BaseModel):
    housing_units_within_1km: Optional[float] = None
    housing_density_per_km2: Optional[float] = None
    nearest_hospital: HospitalRef = Field(default_factory=HospitalRef)
    nearest_school_distance_m: Optional[float] = None


class Structure(BaseModel):
    height_m: Optional[float] = None
    footprint_sqm: Optional[float] = None
    overture_class: Optional[str] = None


class USGSGaugeSummary(BaseModel):
    gage_name: Optional[str] = None
    distance_m: Optional[float] = None
    discharge_cfs: Optional[float] = None
    discharge_class: DischargeClassEnum = "unknown"
    fetched_at: Optional[str] = None


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


class ResponseCard(BaseModel):
    card_id: str
    triggered_by_action_card: str
    generated_at: str
    site: SiteRef

    water_sources: list[WaterSource] = Field(default_factory=list)
    access_routes: list[AccessRoute] = Field(default_factory=list)
    fire_station: FireStation
    nearest_airport: Airport
    hazmat_sites: list[HazmatSite] = Field(default_factory=list)
    environmental_constraints: list[EnvironmentalConstraint] = Field(default_factory=list)
    responsible_agency: Optional[str] = None
    comms: Comms
    evacuation: Evacuation
    structure: Structure
    usgs_gage_summary: USGSGaugeSummary

    citations: list[Citation] = Field(default_factory=list)
    w_sources: list[WSource] = Field(default_factory=list)
    response_card_version: str
