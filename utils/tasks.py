from dataclasses import dataclass


@dataclass(slots=True)
class HistTask:
    data: list
    title: str
    path: str


@dataclass(slots=True)
class OverlayTask:
    a: list
    b: list
    label_a: str
    label_b: str
    title: str
    path: str


@dataclass(slots=True)
class ECDFTask:
    data: list
    title: str
    path: str


@dataclass(slots=True)
class ECDFOverlayTask:
    a: list
    b: list
    label_a: str
    label_b: str
    title: str
    path: str


@dataclass(slots=True)
class StatsTask:
    a: list
    b: list
    title: str
    path: str


@dataclass(slots=True)
class FileTask:
    path: str
    size: int
    compressed: bool = False


@dataclass(slots=True)
class ZipTask:
    zip_path: str
    member: str
    size: int
    compressed: bool
