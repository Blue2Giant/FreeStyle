from dataclasses import dataclass
from uuid import UUID


@dataclass
class DataTrackInfo:
    _source: str
    _url: str
    _key: str | None = None
    _sequence_id: UUID | None = None
    _choice_id: int | None = None
    _loss: float | list[list[float]] | None = None
    _timestep: float | list[list[float]] | None = None
    _target_width: float | list[list[float]] | None = None
    _target_height: float | list[list[float]] | None = None
