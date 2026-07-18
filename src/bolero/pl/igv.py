from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import igv_notebook
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex


def get_colors(
    palette: str = "tab20", n_colors: int = 30, shuffle: bool = False
) -> list[str]:
    """Get a list of colors in hex format from a given palette."""
    colors = sns.color_palette(palette, n_colors=n_colors)
    hex_list = [to_hex(c) for c in colors]
    if shuffle:
        hex_list = np.random.choice(hex_list, size=n_colors, replace=False).tolist()
    return hex_list


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
    type: str | None = None  # Track type - inferred from file format if not specified
    # sourceType: str = "file"  # Type of data source
    format: str | None = None  # File format - inferred from filename if not specified
    url: str | None = None  # URL to the track data resource

    # Indexing properties
    # URL to file index (BAM .bai, tabix .tbi, etc.)
    indexURL: str | None = None
    indexed: bool | None = None  # Explicit flag for non-indexed resources

    # Display properties
    order: int | None = None  # Relative order of track position
    color: str | None = None  # CSS color value for track features
    height: int = 50  # Initial height of track viewport in pixels
    minHeight: int = 50  # Minimum height of track in pixels
    maxHeight: int = 500  # Maximum height of track in pixels

    # Visibility properties
    visibilityWindow: int | None = None  # Maximum window size in base pairs

    # Interaction properties
    removable: bool = True  # Include "remove" item in track menu

    # Authentication properties
    headers: dict[str, str] | None = None  # HTTP headers for requests
    oauthToken: str | Callable[[], str] | None = None  # OAuth token or function

    def __post_init__(self):
        self.url = str(self.url)

    def to_dict(self) -> dict[str, Any]:
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
    type: str | None = "wig"
    autoscale: bool | None = True
    autoscaleGroup: str | None = None
    min: float | None = 0
    max: float | None = None
    altColor: str | None = None
    colorScale: dict[str, Any] | None = None
    guideLines: list[dict[str, Any]] | None = None
    graphType: str | None = "bar"
    flipAxis: bool = False
    windowFunction: str = "mean"

    def set_autoscale(self, enabled: bool = True, group: str | None = None) -> None:
        """Enable or disable autoscaling with optional group identifier"""
        self.autoscale = enabled
        if group is not None:
            self.autoscaleGroup = group

    def set_scale(
        self, min_val: float | None = None, max_val: float | None = None
    ) -> None:
        """Set minimum and maximum values for the data scale"""
        if min_val is not None:
            self.min = min_val
        if max_val is not None:
            self.max = max_val


class Annotation(IGVTrack):
    """
    Annotation track class for displaying genomic features.
    """

    type: str | None = "annotation"
    format: str | None = "bed"
    url: str | None = None
    indexURL: str | None = None


def infer_track_class(config: dict[str, Any]) -> str:
    """Infer the track class from the configuration"""
    track_type = config.get("type")
    if track_type is None:
        url = str(config.get("url")).lower()
        if url is None:
            raise ValueError("URL is not specified")
        if url.endswith(".gz"):
            url = url[:-3]
        suffix = url.split(".")[-1]
        if suffix in ["bigwig", "bw"]:
            return Wig
        elif suffix in ["bed", "gff", "gff3", "gtf", "bedpe"]:
            return Annotation
        else:
            raise ValueError(
                f"Can not infer track type from URL {url} with suffix {suffix}"
            )
    else:
        if track_type == "wig":
            return Wig
        else:
            raise NotImplementedError(f"Track type {track_type} is not implemented")


@dataclass
class Browser:
    """IGV Browser wrapper with enhanced track management"""

    # Required: One of genome or reference must be set
    genome: str | None = None  # String identifier (e.g., "hg19")
    # Object defining reference genome
    reference: dict[str, Any] | None = None

    # Display and UI options
    flanking: int = 1000  # Distance to pad sides of feature on search
    # Initial genomic location(s)
    locus: str | list[str] | None = None
    minimumBases: int = 40  # Minimum window size when zooming in

    # Genome list options
    genomeList: str | list[dict[str, Any]] | None = None  # URL or inline array
    loadDefaultGenomes: bool = True  # Load default genome list

    # Query parameters
    # Support initialization by query parameters
    queryParametersSupported: bool = False

    # Search configuration
    search: dict[str, Any] | None = None  # Search service configuration

    # UI visibility options
    showAllChromosomes: bool = False  # Show all chromosomes in pulldown
    showChromosomeWidget: bool = True  # Show chromosome pulldown control
    showNavigation: bool = True  # Show basic navigation controls
    showIdeogram: bool = True  # Show chromosome ideogram
    showSVGButton: bool = True  # Show SVG save button
    showRuler: bool = True  # Show genomic ruler track
    showCenterGuide: bool = False  # Show center guide lines
    showCursorTrackGuide: bool = True  # Show cursor following line

    # Track configuration
    trackDefaults: dict | None = None  # Default settings for track types
    tracks: list[dict[str, Any]] = field(default_factory=list)  # Initial tracks
    roi: list[dict[str, Any]] = field(default_factory=list)  # Regions of interest

    # Authentication
    oauthToken: str | None = None  # OAuth access token
    apiKey: str | None = None  # Google API key
    clientId: str | None = None  # Google client ID

    # Display options
    nucleotideColors: dict | None = None  # Nucleotide color table
    # Show sample names for supported tracks
    showSampleNames: bool | None = None
    sampleinfo: dict | list[dict] | None = None  # Sample information

    # Internal
    _browser_obj: igv_notebook.Browser | None = None

    def __post_init__(self):
        igv_notebook.init()  # initialize igv notebook

        # Validate that either genome or reference is specified
        if not self.genome and not self.reference:
            raise ValueError(
                "Either 'genome' or 'reference' must be specified in configuration"
            )

        if self.genome and self.reference:
            raise ValueError("Only one of 'genome' or 'reference' can be specified")

    def to_dict(self) -> dict[str, Any]:
        """Convert the browser configuration to a dictionary suitable for IGV."""
        to_return = {}
        for key, value in self.__dict__.items():
            match key:
                case "tracks" | "roi":
                    to_return[key] = [t.to_dict() for t in value]
                case "_browser_obj":
                    pass
                case _:
                    if value is not None:
                        to_return[key] = value
        return to_return

    @property
    def browser(self) -> igv_notebook.Browser:
        """Get the IGV browser object."""
        if self._browser_obj is None:
            self._browser_obj = igv_notebook.Browser(self.to_dict())
        return self._browser_obj

    def load_track(self, **kwargs):
        """Add a track to the browser"""
        track_class = infer_track_class(kwargs)
        valid_keys = track_class.__dataclass_fields__.keys()
        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        self.tracks.append(track_class(**kwargs))

    def load_track_table(self, track_table: pd.DataFrame, **other_config: Any):
        """Load a track table into the browser"""
        for _, row in track_table.iterrows():
            this_config = {**other_config, **row.to_dict()}
            self.load_track(**this_config)
