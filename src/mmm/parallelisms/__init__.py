"""
mmm/parallelisms/__init__.py
"""

from mmm.parallelisms.parallel_dims import ParallelDims
from mmm.parallelisms.parallelize_llama import parallelize_llama
from mmm.parallelisms.pipeline_llama import pipeline_llama

__all__ = [
    'models_parallelize_fns',
    'models_pipelining_fns',
    'ParallelDims',
]

models_parallelize_fns = {
    'llama3': parallelize_llama,
}

models_pipelining_fns = {
    'llama3': pipeline_llama,
}
