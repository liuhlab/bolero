from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl", "init", "print_environments"]

__version__ = version("bolero")

import warnings

from .pp import Genome, Sequence
from .tl.generic.train_helper import hg38_splits, mm10_splits
from .utils import init, print_environments

warnings.filterwarnings("ignore", module="pyranges")
