import pytest
from eval_framework.metrics.rouge import ROUGEMetric

class TestROUGEMetric:
    @pytest.fixture
    def metric(self):
        """Create a ROUGEMetric instance."""
        return ROUGEMetric(
            metrics=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def test_rouge_initialization(self, metric):
        """Test ROUGEMetric initialization."""
        assert set(metric.metrics) == {"rouge1", "rouge2", "rougeL"}
        assert metric.use_stemmer is True

    def test_rouge_computation(self, metric):
        """Test ROUGE score computation."""
        predictions = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy dog"
        ]
        references = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog"
        ]
        
        results = metric.compute(predictions, references)
        
        assert "rouge1" in results
        assert "rouge2" in results
        assert "rougeL" in results
        assert all(0 <= score <= 1 for score in results.values())

    def test_rouge_perfect_match(self, metric):
        """Test ROUGE score with perfect matches."""
        predictions = ["The quick brown fox jumps over the lazy dog"]
        references = ["The quick brown fox jumps over the lazy dog"]
        
        results = metric.compute(predictions, references)
        
        assert results["rouge1"] == 1.0
        assert results["rouge2"] == 1.0
        assert results["rougeL"] == 1.0

    def test_rouge_no_match(self, metric):
        """Test ROUGE score with no matches."""
        predictions = ["The quick brown fox jumps over the lazy dog"]
        references = ["A completely different sentence with no overlap"]
        
        results = metric.compute(predictions, references)
        
        assert results["rouge1"] == 0.0
        assert results["rouge2"] == 0.0
        assert results["rougeL"] == 0.0

    def test_rouge_partial_match(self, metric):
        """Test ROUGE score with partial matches."""
        predictions = ["The quick brown fox jumps over the lazy dog"]
        references = ["The quick brown fox sleeps in the sun"]
        
        results = metric.compute(predictions, references)
        
        assert 0 < results["rouge1"] < 1
        assert 0 < results["rouge2"] < 1
        assert 0 < results["rougeL"] < 1

    def test_rouge_without_stemmer(self):
        """Test ROUGE score computation without stemming."""
        metric = ROUGEMetric(metrics=["rouge1"], use_stemmer=False)
        
        predictions = ["The quick brown foxes jump over the lazy dogs"]
        references = ["The quick brown fox jumps over the lazy dog"]
        
        results = metric.compute(predictions, references)
        
        assert 0 < results["rouge1"] < 1

    def test_rouge_empty_inputs(self, metric):
        """Test ROUGE score computation with empty inputs."""
        results = metric.compute([], [])
        
        assert results["rouge1"] == 0.0
        assert results["rouge2"] == 0.0
        assert results["rougeL"] == 0.0

    def test_rouge_mismatched_lengths(self, metric):
        """Test ROUGE score computation with mismatched input lengths."""
        predictions = ["The quick brown fox"]
        references = ["The quick brown fox", "Another reference"]
        
        with pytest.raises(ValueError):
            metric.compute(predictions, references)

    def test_rouge_get_info(self, metric):
        """Test ROUGEMetric get_info method."""
        info = metric.get_info()
        assert info["name"] == "rouge"
        assert set(info["metrics"]) == {"rouge1", "rouge2", "rougeL"}
        assert info["use_stemmer"] is True

    def test_rouge_custom_metrics(self):
        """Test ROUGE score computation with custom metrics."""
        metric = ROUGEMetric(metrics=["rouge1"])
        
        predictions = ["The quick brown fox jumps over the lazy dog"]
        references = ["The quick brown fox jumps over the lazy dog"]
        
        results = metric.compute(predictions, references)
        
        assert "rouge1" in results
        assert "rouge2" not in results
        assert "rougeL" not in results
        assert results["rouge1"] == 1.0

    def test_rouge_multiple_references(self, metric):
        """Test ROUGE score computation with multiple references."""
        predictions = ["The quick brown fox jumps over the lazy dog"]
        references = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy dog"
        ]
        
        results = metric.compute(predictions, references)
        
        assert all(0 <= score <= 1 for score in results.values()) 