from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import igv_notebook
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex


def get_colors(
    palette: str = "tab20", n_colors: int = 30, shuffle: bool = False
) -> List[str]:
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
    type: Optional[str] = (
        None  # Track type - inferred from file format if not specified
    )
    # sourceType: str = "file"  # Type of data source
    format: Optional[str] = (
        None  # File format - inferred from filename if not specified
    )
    url: Optional[str] = None  # URL to the track data resource

    # Indexing properties
    # URL to file index (BAM .bai, tabix .tbi, etc.)
    indexURL: Optional[str] = None
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

    def __post_init__(self):
        self.url = str(self.url)

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


def infer_track_class(config: Dict[str, Any]) -> str:
    """Infer the track class from the configuration"""
    track_type = config.get("type")
    if track_type is None:
        url = str(config.get("url")).lower()
        if url is None:
            raise ValueError("URL is not specified")
        if url.endswith(".bigwig"):
            return Wig
        elif url.endswith(".bw"):
            return Wig
        else:
            raise ValueError(f"Can not infer track type from URL {url}")
    else:
        if track_type == "wig":
            return Wig
        else:
            raise NotImplementedError(f"Track type {track_type} is not implemented")


@dataclass
class Browser:
    """IGV Browser wrapper with enhanced track management"""

    # Required: One of genome or reference must be set
    genome: Optional[str] = None  # String identifier (e.g., "hg19")
    # Object defining reference genome
    reference: Optional[Dict[str, Any]] = None

    # Display and UI options
    flanking: int = 1000  # Distance to pad sides of feature on search
    # Initial genomic location(s)
    locus: Optional[Union[str, List[str]]] = None
    minimumBases: int = 40  # Minimum window size when zooming in

    # Genome list options
    genomeList: Optional[Union[str, List[Dict[str, Any]]]] = None  # URL or inline array
    loadDefaultGenomes: bool = True  # Load default genome list

    # Query parameters
    # Support initialization by query parameters
    queryParametersSupported: bool = False

    # Search configuration
    search: Optional[Dict[str, Any]] = None  # Search service configuration

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
    trackDefaults: Optional[dict] = None  # Default settings for track types
    tracks: List[Dict[str, Any]] = field(default_factory=list)  # Initial tracks
    roi: List[Dict[str, Any]] = field(default_factory=list)  # Regions of interest

    # Authentication
    oauthToken: Optional[str] = None  # OAuth access token
    apiKey: Optional[str] = None  # Google API key
    clientId: Optional[str] = None  # Google client ID

    # Display options
    nucleotideColors: Optional[dict] = None  # Nucleotide color table
    # Show sample names for supported tracks
    showSampleNames: Optional[bool] = None
    sampleinfo: Optional[Union[dict, List[dict]]] = None  # Sample information

    # Internal
    _browser_obj: Optional[igv_notebook.Browser] = None

    def __post_init__(self):
        igv_notebook.init()  # initialize igv notebook

        # Validate that either genome or reference is specified
        if not self.genome and not self.reference:
            raise ValueError(
                "Either 'genome' or 'reference' must be specified in configuration"
            )

        if self.genome and self.reference:
            raise ValueError("Only one of 'genome' or 'reference' can be specified")

    def to_dict(self) -> Dict[str, Any]:
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
