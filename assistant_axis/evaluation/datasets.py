"""
Dataset loaders for educational evaluation.

Supports:
- ASAP-SAS (Short Answer Scoring) from Kaggle
"""

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from urllib.request import urlretrieve
import os


@dataclass
class ScoringExample:
    """A single scoring example with student answer and human score."""
    id: str
    essay_set: int
    answer_text: str
    score: int
    score2: Optional[int] = None  # Second rater score if available
    question: Optional[str] = None  # The question/prompt if available
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "essay_set": self.essay_set,
            "answer_text": self.answer_text,
            "score": self.score,
            "score2": self.score2,
            "question": self.question,
            "metadata": self.metadata,
        }


# ASAP-SAS Question prompts for each essay set (1-10)
# These are the original questions students were responding to
ASAP_SAS_PROMPTS = {
    1: """After reading the groups procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information.""",

    2: """List and describe three processes used by cells to control the movement of substances across the cell membrane.""",

    3: """Starting with mRNA leaving the nucleus, list and describe four major steps involved in protein synthesis.""",

    4: """A group of students wrote the following procedure for their investigation.

Procedure:
1. Determine the mass of four different rubber balls.
2. Drop each ball from the same height onto a hard surface.
3. Measure how high each ball bounces.

After reading the groups procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information.""",

    5: """Explain how pandas in China areستمرار affected by both abiotic and biotic factors in their ecosystem.""",

    6: """Explain why the__(scientist)__(action) to__(purpose). Use information from the story to support your answer.""",

    7: """Based on the results of the students' investigation, what is the effect of the physical__(characteristic) on the__(result)?""",

    8: """Describe two ways the__(subject) were__(action). Use information from the__(source) to support your answer.""",

    9: """Explain the__(concept) using evidence from the__(source).""",

    10: """Using the information in the passage, explain why__(outcome). Support your answer with details from the passage.""",
}

# Score ranges for each essay set
ASAP_SAS_SCORE_RANGES = {
    1: (0, 3),
    2: (0, 3),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 2),
    8: (0, 2),
    9: (0, 2),
    10: (0, 3),
}


class ASAPSASDataset:
    """
    ASAP-SAS (Short Answer Scoring) Dataset loader.

    The dataset contains ~17,000 short answer responses from students
    on 10 different topics, scored 0-3 (or 0-2, 0-4 for some sets).

    Download from: https://www.kaggle.com/c/asap-sas/data

    Args:
        data_path: Path to the TSV file (train.tsv or test.tsv)
        essay_sets: Optional list of essay sets to include (1-10)
        include_prompts: Whether to include question prompts

    Example:
        dataset = ASAPSASDataset("data/asap-sas/train.tsv")
        for example in dataset:
            print(f"Score: {example.score}, Answer: {example.answer_text[:50]}...")
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        essay_sets: Optional[List[int]] = None,
        include_prompts: bool = True,
    ):
        self.data_path = Path(data_path)
        self.essay_sets = essay_sets
        self.include_prompts = include_prompts
        self.examples: List[ScoringExample] = []

        if self.data_path.exists():
            self._load_data()
        else:
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                f"Download from https://www.kaggle.com/c/asap-sas/data"
            )

    def _load_data(self):
        """Load data from TSV file."""
        with open(self.data_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                essay_set = int(row.get('EssaySet', row.get('essay_set', 0)))

                # Filter by essay set if specified
                if self.essay_sets and essay_set not in self.essay_sets:
                    continue

                # Get score (handle different column names)
                score = row.get('Score1', row.get('score1', row.get('Score', row.get('score'))))
                if score is None or score == '':
                    continue
                score = int(score)

                # Get second score if available
                score2 = row.get('Score2', row.get('score2'))
                score2 = int(score2) if score2 and score2 != '' else None

                # Get answer text
                answer_text = row.get('EssayText', row.get('essay_text', row.get('text', '')))

                # Get ID
                example_id = row.get('Id', row.get('id', str(len(self.examples))))

                # Get prompt if available
                question = None
                if self.include_prompts and essay_set in ASAP_SAS_PROMPTS:
                    question = ASAP_SAS_PROMPTS[essay_set]

                example = ScoringExample(
                    id=str(example_id),
                    essay_set=essay_set,
                    answer_text=answer_text.strip(),
                    score=score,
                    score2=score2,
                    question=question,
                    metadata={
                        "score_range": ASAP_SAS_SCORE_RANGES.get(essay_set, (0, 3)),
                    }
                )
                self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[ScoringExample]:
        return iter(self.examples)

    def __getitem__(self, idx: int) -> ScoringExample:
        return self.examples[idx]

    def sample(self, n: int, seed: Optional[int] = None) -> List[ScoringExample]:
        """
        Get a random sample of examples.

        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled examples
        """
        if seed is not None:
            random.seed(seed)
        return random.sample(self.examples, min(n, len(self.examples)))

    def get_by_essay_set(self, essay_set: int) -> List[ScoringExample]:
        """Get all examples from a specific essay set."""
        return [ex for ex in self.examples if ex.essay_set == essay_set]

    def get_score_distribution(self, essay_set: Optional[int] = None) -> Dict[int, int]:
        """Get the distribution of scores."""
        examples = self.examples
        if essay_set is not None:
            examples = self.get_by_essay_set(essay_set)

        distribution = {}
        for ex in examples:
            distribution[ex.score] = distribution.get(ex.score, 0) + 1
        return dict(sorted(distribution.items()))

    def get_essay_sets(self) -> List[int]:
        """Get list of unique essay sets in the dataset."""
        return sorted(set(ex.essay_set for ex in self.examples))

    def statistics(self) -> Dict:
        """Get dataset statistics."""
        essay_sets = self.get_essay_sets()
        return {
            "total_examples": len(self.examples),
            "essay_sets": essay_sets,
            "examples_per_set": {
                es: len(self.get_by_essay_set(es)) for es in essay_sets
            },
            "score_distribution": self.get_score_distribution(),
            "avg_answer_length": sum(len(ex.answer_text) for ex in self.examples) / len(self.examples) if self.examples else 0,
        }

    def to_json(self, path: Union[str, Path]):
        """Export dataset to JSON."""
        with open(path, 'w') as f:
            json.dump([ex.to_dict() for ex in self.examples], f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ASAPSASDataset":
        """Load dataset from JSON."""
        instance = cls.__new__(cls)
        instance.examples = []

        with open(path, 'r') as f:
            data = json.load(f)

        for item in data:
            instance.examples.append(ScoringExample(**item))

        return instance


def load_asap_sas(
    data_dir: Union[str, Path],
    split: str = "train",
    essay_sets: Optional[List[int]] = None,
) -> ASAPSASDataset:
    """
    Convenience function to load ASAP-SAS dataset.

    Args:
        data_dir: Directory containing the dataset files
        split: "train" or "test"
        essay_sets: Optional list of essay sets to include

    Returns:
        ASAPSASDataset instance
    """
    data_dir = Path(data_dir)

    # Try common file names
    possible_names = [
        f"{split}.tsv",
        f"{split}_rel_2.tsv",
        f"train_rel_2.tsv" if split == "train" else "test_public.tsv",
    ]

    for name in possible_names:
        path = data_dir / name
        if path.exists():
            return ASAPSASDataset(path, essay_sets=essay_sets)

    raise FileNotFoundError(
        f"Could not find {split} data in {data_dir}. "
        f"Tried: {', '.join(possible_names)}"
    )


# Placeholder for future datasets
class DatasetRegistry:
    """Registry for educational datasets."""

    _datasets = {
        "asap-sas": ASAPSASDataset,
    }

    @classmethod
    def available(cls) -> List[str]:
        """List available datasets."""
        return list(cls._datasets.keys())

    @classmethod
    def load(cls, name: str, *args, **kwargs):
        """Load a dataset by name."""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {cls.available()}")
        return cls._datasets[name](*args, **kwargs)
