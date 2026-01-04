# Helpers so pytest (and users) can import and run the example scripts.

from .ex01_global_then_qucb import run as example1
from .ex02_prior_mean import run as example2
from .ex03_local_then_finetune import run as example3
from .ex04_fixed_features import run as example4

__all__ = ["example1", "example2", "example3", "example4"]
