from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import igv_notebook


@dataclass
class IGVTrack:
    """
    IGV Track configuration class with all supported properties.

    This class represents a track configuration for IGV (Integrative Genomics Viewer)
    with proper type hints, defaults, and validation.
    """

    # Required fields
    name: str  # Display name (label). Required

    # Core track properties
    type: Optional[str] = (
        None  # Track type - inferred from file format if not specified
    )
    sourceType: str = "file"  # Type of data source
    format: Optional[str] = (
        None  # File format - inferred from filename if not specified
    )
    url: Optional[str] = None  # URL to the track data resource

    # Indexing properties
    indexURL: Optional[str] = None  # URL to file index (BAM .bai, tabix .tbi, etc.)
    indexed: Optional[bool] = None  # Explicit flag for non-indexed resources

    # Display properties
    order: Optional[int] = None  # Relative order of track position
    color: Optional[str] = None  # CSS color value for track features
    height: int = 50  # Initial height of track viewport in pixels
    minHeight: int = 50  # Minimum height of track in pixels
    maxHeight: int = 500  # Maximum height of track in pixels

    # Visibility properties
    visibilityWindow: Optional[int] = None  # Maximum window size in base pairs

    # Interaction properties
    removable: bool = True  # Include "remove" item in track menu

    # Authentication properties
    headers: Optional[Dict[str, str]] = None  # HTTP headers for requests
    oauthToken: Optional[Union[str, Callable[[], str]]] = (
        None  # OAuth token or function
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the track configuration to a dictionary suitable for IGV.
        Filters out None values and includes extra configuration.
        """
        result = {}

        # Add all non-None attributes
        for key, value in self.__dict__.items():
            if value is not None:
                # Convert enum values to strings
                if isinstance(value, Enum):
                    value = value.value
                result[key] = value

        return result


@dataclass
class Wig(IGVTrack):
    """
    Wig track class for displaying quantitative data as bar charts, line plots, or points.

    Supports wig, bedGraph, and bigwig file formats with extensive configuration options
    for visualization including autoscaling, color schemes, and guide lines.
    """

    # Wig-specific properties
    type: Optional[str] = "wig"
    autoscale: Optional[bool] = True
    autoscaleGroup: Optional[str] = None
    min: Optional[float] = 0
    max: Optional[float] = None
    altColor: Optional[str] = None
    colorScale: Optional[Dict[str, Any]] = None
    guideLines: Optional[List[Dict[str, Any]]] = None
    graphType: Optional[str] = "bar"
    flipAxis: bool = False
    windowFunction: str = "mean"

    def set_autoscale(self, enabled: bool = True, group: Optional[str] = None) -> None:
        """Enable or disable autoscaling with optional group identifier"""
        self.autoscale = enabled
        if group is not None:
            self.autoscaleGroup = group

    def set_scale(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> None:
        """Set minimum and maximum values for the data scale"""
        if min_val is not None:
            self.min = min_val
        if max_val is not None:
            self.max = max_val


class Browser:
    """IGV Browser wrapper with enhanced track management"""

    def __init__(self, genome: str):
        self.genome = genome
        self.tracks = []
        self.browser = igv_notebook.Browser(
            {
                "genome": self.genome,
                "tracks": [],
            }
        )

    def add_track(self, track: IGVTrack | dict[str, IGVTrack]):
        """Add a track to the browser"""
        if isinstance(track, IGVTrack):
            track_dict = track.to_dict()
        else:
            track_dict = track

        self.tracks.append(track_dict)
        self.browser.add_track(track_dict)

    def add_tracks(self, tracks: list):
        """Add multiple tracks to the browser"""
        for track in tracks:
            self.add_track(track)

    def remove_track(self, track_name: str):
        """Remove a track by name"""
        self.tracks = [t for t in self.tracks if t.get("name") != track_name]
        # Note: igv_notebook may not support direct track removal
        # You might need to recreate the browser instance

    def get_track_configs(self) -> list:
        """Get all track configurations as dictionaries"""
        return [
            track.to_dict() if hasattr(track, "to_dict") else track
            for track in self.tracks
        ]

    def show(self):
        """Display the IGV browser"""
        self.browser.show()
