"""
Evaluation runner for steered models on scoring tasks.

Provides a configurable framework for evaluating how different
trait steering configurations affect model scoring performance.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .datasets import ASAPSASDataset, ScoringExample
from .metrics import compute_all_metrics, quadratic_weighted_kappa


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation runs.

    Args:
        traits: List of trait names to use for steering (None = no steering)
        coefficients: Coefficients for each trait (or single value for all)
        use_assistant_axis: If True, use assistant axis instead of traits
        assistant_coefficient: Coefficient for assistant axis steering
        use_capping: If True, apply activation capping
        capping_experiment: Specific capping experiment ID

        max_new_tokens: Max tokens for model generation
        temperature: Sampling temperature (0 = greedy)
        num_samples: Number of examples to evaluate (None = all)
        essay_sets: Specific essay sets to evaluate (None = all)
        seed: Random seed for sampling

        system_prompt_template: Template for system prompt
        user_prompt_template: Template for user prompt
        score_parser: Function to parse score from model output
    """
    # Steering configuration (can be modified at runtime)
    traits: Optional[List[str]] = None
    coefficients: Union[float, List[float]] = 1.0
    use_assistant_axis: bool = False
    assistant_coefficient: float = 0.0
    use_capping: bool = False
    capping_experiment: Optional[str] = None

    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.0

    # Evaluation parameters
    num_samples: Optional[int] = None
    essay_sets: Optional[List[int]] = None
    seed: int = 42

    # Prompt templates
    system_prompt_template: str = """You are an expert educational assessor. Your task is to score student answers on a scale from {min_score} to {max_score}.

Scoring Guidelines:
- {max_score}: Excellent - Complete, accurate, well-explained response
- {mid_score}: Adequate - Partially correct with some missing elements
- {min_score}: Poor - Incorrect, incomplete, or missing response

Respond with ONLY the numeric score ({min_score}-{max_score}), nothing else."""

    user_prompt_template: str = """Question: {question}

Student Answer: {answer}

Score ({min_score}-{max_score}):"""

    def __post_init__(self):
        # Normalize coefficients to list
        if self.traits and isinstance(self.coefficients, (int, float)):
            self.coefficients = [float(self.coefficients)] * len(self.traits)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "traits": self.traits,
            "coefficients": self.coefficients,
            "use_assistant_axis": self.use_assistant_axis,
            "assistant_coefficient": self.assistant_coefficient,
            "use_capping": self.use_capping,
            "capping_experiment": self.capping_experiment,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "num_samples": self.num_samples,
            "essay_sets": self.essay_sets,
            "seed": self.seed,
        }


@dataclass
class EvaluationResult:
    """
    Results from an evaluation run.

    Contains metrics, predictions, and metadata.
    """
    config: EvaluationConfig
    metrics: Dict[str, float]
    predictions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qwk(self) -> float:
        """Get QWK score."""
        return self.metrics.get("qwk", 0.0)

    @property
    def accuracy(self) -> float:
        """Get accuracy score."""
        return self.metrics.get("accuracy", 0.0)

    def summary(self) -> str:
        """Get a summary string."""
        lines = [
            f"=== Evaluation Results ===",
            f"Steering: {self._steering_description()}",
            f"Samples: {len(self.predictions)}",
            f"",
            f"Metrics:",
            f"  QWK:      {self.metrics.get('qwk', 0):.4f}",
            f"  Accuracy: {self.metrics.get('accuracy', 0):.4f}",
            f"  MAE:      {self.metrics.get('mae', 0):.4f}",
            f"  Adjacent: {self.metrics.get('adjacent_agreement', 0):.4f}",
        ]
        return "\n".join(lines)

    def _steering_description(self) -> str:
        if self.config.use_capping:
            return f"Capping ({self.config.capping_experiment})"
        elif self.config.use_assistant_axis:
            return f"Assistant Axis (coeff={self.config.assistant_coefficient})"
        elif self.config.traits:
            return f"Traits: {self.config.traits} (coeffs={self.config.coefficients})"
        else:
            return "None (baseline)"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "predictions": self.predictions,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationResult":
        """Load results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            config=EvaluationConfig(**data["config"]),
            metrics=data["metrics"],
            predictions=data["predictions"],
            metadata=data.get("metadata", {}),
        )


def default_score_parser(text: str, min_score: int = 0, max_score: int = 3) -> Optional[int]:
    """
    Parse a numeric score from model output.

    Handles various formats:
    - Plain number: "2"
    - With explanation: "Score: 2" or "2 - good answer"
    - In brackets: "[2]" or "(2)"

    Args:
        text: Model output text
        min_score: Minimum valid score
        max_score: Maximum valid score

    Returns:
        Parsed score or None if parsing fails
    """
    text = text.strip()

    # Try to find a number at the start
    match = re.match(r'^[\[\(]?(\d+)[\]\)]?', text)
    if match:
        score = int(match.group(1))
        if min_score <= score <= max_score:
            return score

    # Try to find "score: X" or "Score: X"
    match = re.search(r'[Ss]core[:\s]+(\d+)', text)
    if match:
        score = int(match.group(1))
        if min_score <= score <= max_score:
            return score

    # Try to find any number in the valid range
    numbers = re.findall(r'\d+', text)
    for num_str in numbers:
        score = int(num_str)
        if min_score <= score <= max_score:
            return score

    return None


class ScoringEvaluator:
    """
    Evaluator for scoring tasks with configurable trait steering.

    This class provides a flexible framework for evaluating how different
    steering configurations affect model performance on scoring tasks.

    Example:
        from assistant_axis import TraitSteerer
        from assistant_axis.evaluation import ScoringEvaluator, ASAPSASDataset

        steerer = TraitSteerer("Qwen/Qwen3-32B")
        dataset = ASAPSASDataset("data/asap-sas/train.tsv")

        evaluator = ScoringEvaluator(steerer)

        # Baseline evaluation (no steering)
        baseline = evaluator.evaluate(dataset, num_samples=100)

        # With trait steering
        evaluator.set_traits(["patient", "thorough"], coefficients=[-2.0, -1.5])
        steered = evaluator.evaluate(dataset, num_samples=100)

        # Compare results
        print(f"Baseline QWK: {baseline.qwk:.4f}")
        print(f"Steered QWK:  {steered.qwk:.4f}")
    """

    def __init__(
        self,
        steerer,  # TraitSteerer instance
        config: Optional[EvaluationConfig] = None,
        score_parser: Optional[Callable] = None,
        verbose: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            steerer: TraitSteerer instance (from trait_steering module)
            config: Evaluation configuration (or use defaults)
            score_parser: Custom function to parse scores from model output
            verbose: Whether to print progress
        """
        self.steerer = steerer
        self.config = config or EvaluationConfig()
        self.score_parser = score_parser or default_score_parser
        self.verbose = verbose

    def set_traits(
        self,
        traits: Optional[List[str]],
        coefficients: Union[float, List[float]] = 1.0,
    ):
        """
        Set traits for steering.

        Args:
            traits: List of trait names (None to disable)
            coefficients: Coefficient(s) for steering
        """
        self.config.traits = traits
        self.config.coefficients = coefficients
        if traits and isinstance(coefficients, (int, float)):
            self.config.coefficients = [float(coefficients)] * len(traits)

    def set_assistant_steering(self, coefficient: float):
        """
        Enable assistant axis steering.

        Args:
            coefficient: Steering coefficient
        """
        self.config.use_assistant_axis = True
        self.config.assistant_coefficient = coefficient
        self.config.traits = None

    def set_capping(self, enabled: bool = True, experiment_id: Optional[str] = None):
        """
        Enable/disable activation capping.

        Args:
            enabled: Whether to enable capping
            experiment_id: Specific capping experiment ID
        """
        self.config.use_capping = enabled
        self.config.capping_experiment = experiment_id

    def clear_steering(self):
        """Clear all steering settings (return to baseline)."""
        self.config.traits = None
        self.config.coefficients = 1.0
        self.config.use_assistant_axis = False
        self.config.assistant_coefficient = 0.0
        self.config.use_capping = False
        self.config.capping_experiment = None

    def _build_prompts(
        self,
        example: ScoringExample,
    ) -> Tuple[str, str]:
        """Build system and user prompts for an example."""
        score_range = example.metadata.get("score_range", (0, 3))
        min_score, max_score = score_range
        mid_score = (min_score + max_score) // 2

        system = self.config.system_prompt_template.format(
            min_score=min_score,
            max_score=max_score,
            mid_score=mid_score,
        )

        question = example.question or "Answer the question based on the provided context."

        user = self.config.user_prompt_template.format(
            question=question,
            answer=example.answer_text,
            min_score=min_score,
            max_score=max_score,
        )

        return system, user

    def _get_steering_context(self):
        """Get the appropriate steering context manager."""
        if self.config.use_capping:
            return self.steerer.cap_activations(
                experiment_id=self.config.capping_experiment
            )
        elif self.config.use_assistant_axis:
            return self.steerer.steer_assistant(
                coefficient=self.config.assistant_coefficient
            )
        elif self.config.traits:
            return self.steerer.steer(
                self.config.traits,
                coefficients=self.config.coefficients,
            )
        else:
            # No steering - use a dummy context manager
            return _DummyContext()

    def score_example(self, example: ScoringExample) -> Tuple[Optional[int], str]:
        """
        Score a single example.

        Args:
            example: The example to score

        Returns:
            Tuple of (predicted_score, raw_output)
        """
        system, user = self._build_prompts(example)
        score_range = example.metadata.get("score_range", (0, 3))

        with self._get_steering_context():
            output = self.steerer.generate(
                user,
                system_prompt=system,
                max_new_tokens=self.config.max_new_tokens,
            )

        predicted = self.score_parser(output, score_range[0], score_range[1])
        return predicted, output

    def evaluate(
        self,
        dataset: ASAPSASDataset,
        num_samples: Optional[int] = None,
        essay_sets: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Run evaluation on a dataset.

        Args:
            dataset: The dataset to evaluate on
            num_samples: Number of samples (overrides config)
            essay_sets: Essay sets to include (overrides config)
            seed: Random seed (overrides config)

        Returns:
            EvaluationResult with metrics and predictions
        """
        # Use provided args or fall back to config
        num_samples = num_samples or self.config.num_samples
        essay_sets = essay_sets or self.config.essay_sets
        seed = seed if seed is not None else self.config.seed

        # Filter by essay set if specified
        examples = list(dataset)
        if essay_sets:
            examples = [ex for ex in examples if ex.essay_set in essay_sets]

        # Sample if needed
        if num_samples and num_samples < len(examples):
            import random
            random.seed(seed)
            examples = random.sample(examples, num_samples)

        # Run evaluation
        predictions = []
        y_true = []
        y_pred = []
        parse_failures = 0

        start_time = time.time()

        for i, example in enumerate(examples):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Processing {i + 1}/{len(examples)}...")

            predicted_score, raw_output = self.score_example(example)

            pred_record = {
                "id": example.id,
                "essay_set": example.essay_set,
                "true_score": example.score,
                "predicted_score": predicted_score,
                "raw_output": raw_output,
                "answer_text": example.answer_text[:200],  # Truncate for storage
            }
            predictions.append(pred_record)

            if predicted_score is not None:
                y_true.append(example.score)
                y_pred.append(predicted_score)
            else:
                parse_failures += 1

        elapsed = time.time() - start_time

        # Compute metrics
        if y_true and y_pred:
            metrics = compute_all_metrics(y_true, y_pred)
        else:
            metrics = {"qwk": 0.0, "accuracy": 0.0, "mae": 0.0}

        metrics["parse_failures"] = parse_failures
        metrics["parse_failure_rate"] = parse_failures / len(examples) if examples else 0

        # Build result
        result = EvaluationResult(
            config=EvaluationConfig(**self.config.to_dict()),  # Copy config
            metrics=metrics,
            predictions=predictions,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "model_name": self.steerer.model_name,
                "dataset_size": len(examples),
            }
        )

        if self.verbose:
            print(result.summary())

        return result


class _DummyContext:
    """Dummy context manager for no-steering case."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def run_evaluation(
    steerer,
    dataset: ASAPSASDataset,
    configs: List[EvaluationConfig],
    verbose: bool = True,
) -> List[EvaluationResult]:
    """
    Run multiple evaluation configurations.

    Convenience function to compare different steering setups.

    Args:
        steerer: TraitSteerer instance
        dataset: Dataset to evaluate on
        configs: List of configurations to test
        verbose: Whether to print progress

    Returns:
        List of evaluation results

    Example:
        configs = [
            EvaluationConfig(),  # Baseline
            EvaluationConfig(traits=["patient"], coefficients=[-2.0]),
            EvaluationConfig(use_assistant_axis=True, assistant_coefficient=1.0),
        ]
        results = run_evaluation(steerer, dataset, configs)
    """
    evaluator = ScoringEvaluator(steerer, verbose=verbose)
    results = []

    for i, config in enumerate(configs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running config {i + 1}/{len(configs)}")
            print(f"{'='*60}")

        evaluator.config = config
        result = evaluator.evaluate(dataset)
        results.append(result)

    return results


def compare_results(results: List[EvaluationResult]) -> str:
    """
    Generate a comparison table for multiple results.

    Args:
        results: List of evaluation results

    Returns:
        Formatted comparison string
    """
    lines = [
        "| Configuration | QWK | Accuracy | MAE | Adjacent |",
        "|---------------|-----|----------|-----|----------|",
    ]

    for result in results:
        desc = result._steering_description()[:30]
        lines.append(
            f"| {desc:<13} | {result.metrics.get('qwk', 0):.3f} | "
            f"{result.metrics.get('accuracy', 0):.3f}    | "
            f"{result.metrics.get('mae', 0):.2f} | "
            f"{result.metrics.get('adjacent_agreement', 0):.3f}    |"
        )

    return "\n".join(lines)
