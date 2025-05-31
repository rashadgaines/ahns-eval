"""Error analysis implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class ErrorAnalysisConfig:
    """Configuration for error analysis."""
    # Clustering configuration
    min_samples: int = 3
    eps: float = 0.5
    min_cluster_size: int = 2
    
    # Feature extraction
    max_features: int = 1000
    ngram_range: Tuple[int, int] = (1, 2)
    
    # Semantic analysis
    similarity_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    device: Optional[str] = None

class ErrorAnalyzer:
    """Error analysis implementation.
    
    This class provides tools for analyzing model failures and error patterns,
    including error clustering, pattern detection, and failure case analysis.
    """
    
    def __init__(
        self,
        config: Optional[ErrorAnalysisConfig] = None,
        **kwargs: Any
    ):
        """Initialize error analyzer.
        
        Args:
            config: Analysis configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or ErrorAnalysisConfig(**kwargs)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize required models."""
        try:
            # Initialize semantic similarity model
            self.similarity_model = SentenceTransformer(
                self.config.similarity_model,
                device=self.config.device
            )
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    def analyze_errors(
        self,
        inputs: List[str],
        predictions: List[str],
        references: List[str],
        metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze model errors and failure patterns.
        
        Args:
            inputs: Input texts
            predictions: Model predictions
            references: Ground truth references
            metrics: Evaluation metrics for each sample
            
        Returns:
            Dictionary containing error analysis results
        """
        # Identify failure cases
        failures = self._identify_failures(inputs, predictions, references, metrics)
        
        # Analyze error patterns
        patterns = self._analyze_patterns(failures)
        
        # Cluster similar errors
        clusters = self._cluster_errors(failures)
        
        # Analyze error types
        error_types = self._analyze_error_types(failures)
        
        return {
            "failures": failures,
            "patterns": patterns,
            "clusters": clusters,
            "error_types": error_types
        }
    
    def _identify_failures(
        self,
        inputs: List[str],
        predictions: List[str],
        references: List[str],
        metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify failure cases based on metrics.
        
        Args:
            inputs: Input texts
            predictions: Model predictions
            references: Ground truth references
            metrics: Evaluation metrics for each sample
            
        Returns:
            List of failure cases with details
        """
        failures = []
        
        for i, (inp, pred, ref, metric) in enumerate(
            zip(inputs, predictions, references, metrics)
        ):
            # Check if this is a failure case
            if not self._is_success(metric):
                failures.append({
                    "index": i,
                    "input": inp,
                    "prediction": pred,
                    "reference": ref,
                    "metrics": metric,
                    "error_type": self._classify_error_type(pred, ref, metric)
                })
        
        return failures
    
    def _is_success(self, metrics: Dict[str, Any]) -> bool:
        """Determine if a prediction is successful based on metrics.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            True if prediction is successful, False otherwise
        """
        # Check various metrics
        if "accuracy" in metrics and metrics["accuracy"] < 1.0:
            return False
        
        if "bleu" in metrics and metrics["bleu"]["bleu"] < 0.7:
            return False
        
        if "rouge" in metrics:
            rouge_scores = metrics["rouge"]
            if any(score["fmeasure"] < 0.7 for score in rouge_scores.values()):
                return False
        
        if "semantic_similarity" in metrics:
            if metrics["semantic_similarity"]["similarity"] < self.config.similarity_threshold:
                return False
        
        return True
    
    def _classify_error_type(
        self,
        prediction: str,
        reference: str,
        metrics: Dict[str, Any]
    ) -> str:
        """Classify the type of error.
        
        Args:
            prediction: Model prediction
            reference: Ground truth reference
            metrics: Evaluation metrics
            
        Returns:
            Error type classification
        """
        # Check for different error types
        if len(prediction.split()) < len(reference.split()) * 0.5:
            return "truncation"
        
        if len(prediction.split()) > len(reference.split()) * 1.5:
            return "repetition"
        
        if "semantic_similarity" in metrics:
            if metrics["semantic_similarity"]["similarity"] < 0.3:
                return "semantic_error"
        
        if "bleu" in metrics and metrics["bleu"]["bleu"] < 0.3:
            return "syntax_error"
        
        return "other"
    
    def _analyze_patterns(
        self,
        failures: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns in failure cases.
        
        Args:
            failures: List of failure cases
            
        Returns:
            Dictionary containing pattern analysis results
        """
        # Extract features
        texts = [f["input"] for f in failures]
        features = self.vectorizer.fit_transform(texts)
        
        # Analyze error type distribution
        error_types = Counter(f["error_type"] for f in failures)
        
        # Analyze common words in failures
        words = " ".join(texts).lower().split()
        word_freq = Counter(words)
        
        # Analyze input lengths
        lengths = [len(t.split()) for t in texts]
        
        return {
            "error_type_distribution": dict(error_types),
            "common_words": dict(word_freq.most_common(20)),
            "length_statistics": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": min(lengths),
                "max": max(lengths)
            },
            "feature_importance": dict(zip(
                self.vectorizer.get_feature_names_out(),
                features.mean(axis=0).A1
            ))
        }
    
    def _cluster_errors(
        self,
        failures: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cluster similar error cases.
        
        Args:
            failures: List of failure cases
            
        Returns:
            Dictionary containing clustering results
        """
        # Get embeddings
        texts = [f["input"] for f in failures]
        embeddings = self.similarity_model.encode(texts)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples
        ).fit(embeddings)
        
        # Organize clusters
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label != -1:  # Skip noise points
                clusters[label].append(failures[i])
        
        # Filter small clusters
        clusters = {
            k: v for k, v in clusters.items()
            if len(v) >= self.config.min_cluster_size
        }
        
        # Analyze clusters
        cluster_analysis = {}
        for label, cases in clusters.items():
            # Get cluster center
            cluster_embeddings = embeddings[
                [i for i, l in enumerate(clustering.labels_) if l == label]
            ]
            center = cluster_embeddings.mean(axis=0)
            
            # Find representative case
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            representative = cases[np.argmin(distances)]
            
            cluster_analysis[label] = {
                "size": len(cases),
                "error_types": Counter(c["error_type"] for c in cases),
                "representative_case": representative
            }
        
        return {
            "clusters": dict(clusters),
            "analysis": cluster_analysis,
            "noise_points": [
                failures[i] for i, label in enumerate(clustering.labels_)
                if label == -1
            ]
        }
    
    def _analyze_error_types(
        self,
        failures: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze different types of errors.
        
        Args:
            failures: List of failure cases
            
        Returns:
            Dictionary containing error type analysis
        """
        # Group failures by error type
        by_type = defaultdict(list)
        for failure in failures:
            by_type[failure["error_type"]].append(failure)
        
        # Analyze each error type
        analysis = {}
        for error_type, cases in by_type.items():
            # Get metrics for this error type
            metrics = [case["metrics"] for case in cases]
            
            # Compute statistics
            analysis[error_type] = {
                "count": len(cases),
                "percentage": len(cases) / len(failures),
                "average_metrics": {
                    metric: np.mean([m[metric] for m in metrics if metric in m])
                    for metric in metrics[0].keys()
                },
                "example_cases": cases[:3]  # Show first 3 examples
            }
        
        return analysis 