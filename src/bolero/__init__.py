from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl"]

__version__ = version("bolero")

import warnings

from .pp import Genome, Sequence

warnings.filterwarnings("ignore", module="pyranges")
