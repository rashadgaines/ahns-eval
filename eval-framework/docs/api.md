# API Documentation

## Core Components

### EvaluationEngine

The main entry point for running evaluations.

```python
from eval_framework.core.engine import EvaluationEngine

engine = EvaluationEngine()
```

#### Methods

- `evaluate(config: dict, progress_callback: Callable = None) -> dict`
  - Runs evaluation with given configuration
  - Returns results dictionary with metrics and predictions

- `register_model(model_type: str, model_class: Type) -> None`
  - Registers a custom model class
  - Used for adding new model types

- `register_evaluator(evaluator_type: str, evaluator_class: Type) -> None`
  - Registers a custom evaluator class
  - Used for adding new evaluation methods

- `register_metric(metric_type: str, metric_class: Type) -> None`
  - Registers a custom metric class
  - Used for adding new metrics

- `save_results(results: dict, output_path: str) -> None`
  - Saves evaluation results to specified path
  - Supports multiple output formats

## Models

### TextModel

Base class for text-based models.

```python
from eval_framework.models.text_model import TextModel

class CustomTextModel(TextModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def predict(self, inputs: List[str]) -> List[str]:
        # Implementation
        pass
```

### MultimodalModel

Base class for multimodal models.

```python
from eval_framework.models.multimodal_model import MultimodalModel

class CustomMultimodalModel(MultimodalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation
        pass
```

## Datasets

### TextDataset

Base class for text datasets.

```python
from eval_framework.datasets.text_dataset import TextDataset

class CustomTextDataset(TextDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_data(self) -> Tuple[List[str], List[str]]:
        # Implementation
        pass
```

### MultimodalDataset

Base class for multimodal datasets.

```python
from eval_framework.datasets.multimodal_dataset import MultimodalDataset

class CustomMultimodalDataset(MultimodalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Implementation
        pass
```

## Evaluators

### BaseEvaluator

Base class for all evaluators.

```python
from eval_framework.core.base import BaseEvaluator

class CustomEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def evaluate(self, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        # Implementation
        pass
```

### ExactMatchEvaluator

Evaluates exact matches between predictions and references.

```python
from eval_framework.evaluators.exact_match import ExactMatchEvaluator

evaluator = ExactMatchEvaluator(
    normalize_text=True,
    case_sensitive=False
)
```

### ROUGEEvaluator

Evaluates using ROUGE metrics.

```python
from eval_framework.evaluators.rouge import ROUGEEvaluator

evaluator = ROUGEEvaluator(
    metrics=["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)
```

## Metrics

### BaseMetric

Base class for all metrics.

```python
from eval_framework.core.base import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute(self, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        # Implementation
        pass
```

### ROUGEMetric

Computes ROUGE scores.

```python
from eval_framework.metrics.rouge import ROUGEMetric

metric = ROUGEMetric(
    metrics=["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)
```

### CLIPScore

Computes CLIP scores for multimodal evaluation.

```python
from eval_framework.metrics.clip_score import CLIPScore

metric = CLIPScore(
    model_name="clip-vit-base-patch32"
)
```

## Configuration

### Model Configuration

```python
model_config = {
    "type": "text",  # or "multimodal"
    "name": "gpt2",
    "batch_size": 4,
    "device": "cpu",  # or "cuda:0"
    "parameters": {
        # Model-specific parameters
    }
}
```

### Dataset Configuration

```python
dataset_config = {
    "type": "text",  # or "multimodal"
    "name": "test_dataset",
    "split": "test",
    "max_samples": 100,
    "batch_size": 4,
    "parameters": {
        # Dataset-specific parameters
    }
}
```

### Evaluator Configuration

```python
evaluator_config = {
    "type": "exact_match",
    "parameters": {
        "normalize_text": True,
        "case_sensitive": False
    }
}
```

### Output Configuration

```python
output_config = {
    "formats": ["json", "html"],
    "save_predictions": True,
    "save_metrics": True,
    "save_error_analysis": True,
    "output_dir": "output/results"
}
```

## Results Format

### Text Evaluation Results

```python
{
    "metrics": {
        "rouge1": 0.75,
        "rouge2": 0.65,
        "rougeL": 0.70
    },
    "predictions": ["pred1", "pred2", ...],
    "references": ["ref1", "ref2", ...],
    "metadata": {
        "model": "gpt2",
        "dataset": "test_dataset",
        "timestamp": "2024-03-20T12:00:00"
    }
}
```

### Multimodal Evaluation Results

```python
{
    "metrics": {
        "clip_score": 0.85,
        "bleu": 0.75
    },
    "predictions": {
        "text": ["pred1", "pred2", ...],
        "image": ["img1", "img2", ...]
    },
    "references": {
        "text": ["ref1", "ref2", ...],
        "image": ["img1", "img2", ...]
    },
    "metadata": {
        "model": "clip-vit-base-patch32",
        "dataset": "coco_captions",
        "timestamp": "2024-03-20T12:00:00"
    }
}
``` 