"""Dataset registry implementation for the evaluation framework."""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import requests
from tqdm import tqdm

from eval_framework.datasets.base import BaseDataset
from eval_framework.datasets.preprocessors import PreprocessingConfig

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Types of datasets supported by the framework."""
    TEXT = "text"
    MULTIMODAL = "multimodal"
    MULTIPLE_CHOICE = "multiple_choice"
    CODE = "code"
    MATH = "math"

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    type: DatasetType
    description: str
    url: Optional[str] = None
    local_path: Optional[Path] = None
    huggingface_id: Optional[str] = None
    preprocessing: Optional[PreprocessingConfig] = None
    metadata: Optional[Dict[str, Any]] = None

class DatasetRegistry:
    """Registry for managing datasets and their configurations."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize dataset registry.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".eval_framework" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._datasets: Dict[str, DatasetConfig] = {}
        self._dataset_classes: Dict[str, Type[BaseDataset]] = {}
        
        # Register built-in datasets
        self._register_builtin_datasets()
    
    def _register_builtin_datasets(self) -> None:
        """Register built-in dataset configurations."""
        # MMLU (Massive Multitask Language Understanding)
        self.register_dataset(
            DatasetConfig(
                name="mmlu",
                type=DatasetType.MULTIPLE_CHOICE,
                description="Massive Multitask Language Understanding benchmark",
                huggingface_id="cais/mmlu",
                preprocessing=PreprocessingConfig(
                    max_length=512,
                    truncation=True,
                    padding=True
                ),
                metadata={
                    "subjects": [
                        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
                        "clinical_knowledge", "college_biology", "college_chemistry",
                        "college_computer_science", "college_mathematics",
                        "college_medicine", "college_physics", "computer_security",
                        "conceptual_physics", "econometrics", "electrical_engineering",
                        "elementary_mathematics", "formal_logic", "global_facts",
                        "high_school_biology", "high_school_chemistry",
                        "high_school_computer_science", "high_school_european_history",
                        "high_school_geography", "high_school_government_and_politics",
                        "high_school_macroeconomics", "high_school_mathematics",
                        "high_school_microeconomics", "high_school_physics",
                        "high_school_psychology", "high_school_statistics",
                        "high_school_us_history", "high_school_world_history",
                        "human_aging", "human_sexuality", "international_law",
                        "jurisprudence", "logical_fallacies", "machine_learning",
                        "management", "marketing", "medical_genetics", "miscellaneous",
                        "moral_disputes", "moral_scenarios", "nutrition",
                        "philosophy", "prehistory", "professional_accounting",
                        "professional_law", "professional_medicine",
                        "professional_psychology", "public_relations",
                        "security_studies", "sociology", "us_foreign_policy",
                        "virology", "world_religions"
                    ]
                }
            )
        )
        
        # HellaSwag
        self.register_dataset(
            DatasetConfig(
                name="hellaswag",
                type=DatasetType.MULTIPLE_CHOICE,
                description="HellaSwag: A Challenge Dataset for Commonsense NLI",
                huggingface_id="Rowan/hellaswag",
                preprocessing=PreprocessingConfig(
                    max_length=512,
                    truncation=True,
                    padding=True
                )
            )
        )
        
        # TruthfulQA
        self.register_dataset(
            DatasetConfig(
                name="truthfulqa",
                type=DatasetType.MULTIPLE_CHOICE,
                description="TruthfulQA: Measuring How Models Mimic Human Falsehoods",
                huggingface_id="truthful_qa",
                preprocessing=PreprocessingConfig(
                    max_length=512,
                    truncation=True,
                    padding=True
                )
            )
        )
        
        # GSM8K
        self.register_dataset(
            DatasetConfig(
                name="gsm8k",
                type=DatasetType.MATH,
                description="Grade School Math 8K",
                huggingface_id="gsm8k",
                preprocessing=PreprocessingConfig(
                    max_length=512,
                    truncation=True,
                    padding=True
                )
            )
        )
        
        # HumanEval
        self.register_dataset(
            DatasetConfig(
                name="humaneval",
                type=DatasetType.CODE,
                description="HumanEval: Hand-Written Evaluation Set",
                huggingface_id="openai_humaneval",
                preprocessing=PreprocessingConfig(
                    max_length=1024,
                    truncation=True,
                    padding=True
                )
            )
        )
        
        # COCO
        self.register_dataset(
            DatasetConfig(
                name="coco",
                type=DatasetType.MULTIMODAL,
                description="Common Objects in Context",
                url="https://download.openmmlab.com/mmediting/data/coco_2017.zip",
                preprocessing=PreprocessingConfig(
                    image_size=(224, 224),
                    normalize=True,
                    max_length=512,
                    truncation=True,
                    padding=True
                )
            )
        )
    
    def register_dataset(
        self,
        config: DatasetConfig,
        dataset_class: Optional[Type[BaseDataset]] = None
    ) -> None:
        """Register a dataset configuration.
        
        Args:
            config: Dataset configuration
            dataset_class: Optional dataset class implementation
        """
        if config.name in self._datasets:
            logger.warning(f"Overwriting existing dataset: {config.name}")
        
        self._datasets[config.name] = config
        if dataset_class is not None:
            self._dataset_classes[config.name] = dataset_class
    
    def get_dataset_config(self, name: str) -> DatasetConfig:
        """Get dataset configuration.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset configuration
            
        Raises:
            KeyError: If dataset is not registered
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset not registered: {name}")
        return self._datasets[name]
    
    def get_dataset_class(self, name: str) -> Type[BaseDataset]:
        """Get dataset class.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset class
            
        Raises:
            KeyError: If dataset class is not registered
        """
        if name not in self._dataset_classes:
            raise KeyError(f"Dataset class not registered: {name}")
        return self._dataset_classes[name]
    
    def list_datasets(self) -> List[str]:
        """List registered datasets.
        
        Returns:
            List of dataset names
        """
        return list(self._datasets.keys())
    
    def download_dataset(self, name: str, force: bool = False) -> Path:
        """Download dataset if not already cached.
        
        Args:
            name: Dataset name
            force: Force download even if cached
            
        Returns:
            Path to downloaded dataset
            
        Raises:
            KeyError: If dataset is not registered
            ValueError: If download fails
        """
        config = self.get_dataset_config(name)
        
        # Check if already downloaded
        if config.local_path and config.local_path.exists() and not force:
            return config.local_path
        
        # Create dataset directory
        dataset_dir = self.cache_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if config.huggingface_id:
                # Download from Hugging Face
                from datasets import load_dataset
                dataset = load_dataset(config.huggingface_id)
                dataset.save_to_disk(dataset_dir)
                
            elif config.url:
                # Download from URL
                response = requests.get(config.url, stream=True)
                response.raise_for_status()
                
                # Get file size
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                file_path = dataset_dir / Path(config.url).name
                with open(file_path, 'wb') as f, tqdm(
                    desc=f"Downloading {name}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True
                ) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)
                
                # Extract if zip file
                if file_path.suffix == '.zip':
                    import zipfile
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    file_path.unlink()  # Remove zip file
                
            else:
                raise ValueError(f"No download source specified for dataset: {name}")
            
            # Update local path
            config.local_path = dataset_dir
            return dataset_dir
            
        except Exception as e:
            raise ValueError(f"Error downloading dataset {name}: {str(e)}")
    
    def get_dataset(
        self,
        name: str,
        split: Optional[str] = None,
        **kwargs: Any
    ) -> BaseDataset:
        """Get dataset instance.
        
        Args:
            name: Dataset name
            split: Dataset split
            **kwargs: Additional dataset parameters
            
        Returns:
            Dataset instance
            
        Raises:
            KeyError: If dataset is not registered
            ValueError: If dataset cannot be loaded
        """
        config = self.get_dataset_config(name)
        
        # Download if needed
        if not config.local_path or not config.local_path.exists():
            self.download_dataset(name)
        
        # Get dataset class
        if name in self._dataset_classes:
            dataset_class = self._dataset_classes[name]
        else:
            # Use default class based on type
            if config.type == DatasetType.MULTIMODAL:
                from eval_framework.datasets.multimodal_dataset import MultimodalDataset
                dataset_class = MultimodalDataset
            else:
                from eval_framework.datasets.json_dataset import JsonDataset
                dataset_class = JsonDataset
        
        # Create dataset instance
        return dataset_class(
            path=config.local_path,
            split=split,
            **kwargs
        ) 