from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl"]

__version__ = version("bolero")

import warnings

from .pp import Genome, Sequence
from .utils import init
from .tl.generic.train_helper import mm10_splits, hg38_splits

warnings.filterwarnings("ignore", module="pyranges")
