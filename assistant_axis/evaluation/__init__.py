"""
Evaluation framework for steered models.

This module provides tools for evaluating steered models on educational datasets
like ASAP-SAS (Short Answer Scoring).

Example:
    from assistant_axis import TraitSteerer
    from assistant_axis.evaluation import (
        ASAPSASDataset,
        ScoringEvaluator,
        quadratic_weighted_kappa,
    )

    # Load dataset
    dataset = ASAPSASDataset("path/to/train.tsv")

    # Create evaluator with configurable traits
    evaluator = ScoringEvaluator(
        steerer=TraitSteerer("Qwen/Qwen3-32B"),
        traits=["patient", "thorough"],
        coefficients=[-2.0, -1.5],
    )

    # Run evaluation
    results = evaluator.evaluate(dataset, num_samples=100)
    print(f"QWK: {results['qwk']:.4f}")
"""

from .datasets import ASAPSASDataset, load_asap_sas
from .metrics import (
    quadratic_weighted_kappa,
    linear_weighted_kappa,
    accuracy,
    confusion_matrix,
)
from .runner import (
    ScoringEvaluator,
    EvaluationConfig,
    EvaluationResult,
    run_evaluation,
)

__all__ = [
    # Datasets
    "ASAPSASDataset",
    "load_asap_sas",
    # Metrics
    "quadratic_weighted_kappa",
    "linear_weighted_kappa",
    "accuracy",
    "confusion_matrix",
    # Runner
    "ScoringEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "run_evaluation",
]
