from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class LikelihoodsEvaluatorConfig(EvaluatorConfig):
    """

    Config class for the evaluation of the coherences module.

    Args:

        batch_size (int) : The batch size to use in the evaluation. Default to 512.
        num_samples (int) : How many samples to use for likelihoods estimates. Default to 1000.
        batch_size_k (int) : How to batch the K samples for likelihoods estimates. Default to 100.
        unified_implementation (bool) : When the paper implementation of the likelihood differ from
            the unified implementation, specify which to use. Default to True.


    """

    num_samples: int = 1000
    batch_size_k: int = 100
    unified_implementation: bool = True