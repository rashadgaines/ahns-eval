import pytest
from unittest.mock import Mock, patch
from eval_framework.models.text_model import TextModel

class TestTextModel:
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.generate.return_value = ["Generated text"]
        return model

    @pytest.fixture
    def text_model(self, mock_model):
        """Create a TextModel instance with a mock model."""
        return TextModel(
            model=mock_model,
            name="test_model",
            batch_size=4,
            device="cpu"
        )

    def test_text_model_initialization(self, text_model):
        """Test TextModel initialization."""
        assert text_model.name == "test_model"
        assert text_model.batch_size == 4
        assert text_model.device == "cpu"

    def test_text_model_load(self, text_model):
        """Test TextModel load method."""
        text_model.load()
        # Verify model is loaded (no assertions needed as it's a mock)

    def test_text_model_predict(self, text_model, mock_model):
        """Test TextModel predict method."""
        inputs = ["Test input"]
        predictions = text_model.predict(inputs)
        
        mock_model.generate.assert_called_once()
        assert predictions == ["Generated text"]

    def test_text_model_predict_batch(self, text_model, mock_model):
        """Test TextModel predict method with batch processing."""
        inputs = ["Input 1", "Input 2", "Input 3", "Input 4", "Input 5"]
        predictions = text_model.predict(inputs)
        
        # Should be called twice due to batch size of 4
        assert mock_model.generate.call_count == 2
        assert len(predictions) == 5

    def test_text_model_get_info(self, text_model):
        """Test TextModel get_info method."""
        info = text_model.get_info()
        assert info["name"] == "test_model"
        assert info["type"] == "text"
        assert info["batch_size"] == 4
        assert info["device"] == "cpu"

    @patch("eval_framework.models.text_model.AutoModelForCausalLM")
    def test_text_model_from_pretrained(self, mock_auto_model):
        """Test creating TextModel from pretrained model."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        
        model = TextModel.from_pretrained(
            model_name="gpt2",
            batch_size=4,
            device="cpu"
        )
        
        assert isinstance(model, TextModel)
        mock_auto_model.from_pretrained.assert_called_once_with("gpt2")

    def test_text_model_error_handling(self, text_model, mock_model):
        """Test TextModel error handling."""
        mock_model.generate.side_effect = Exception("Model error")
        
        with pytest.raises(Exception) as exc_info:
            text_model.predict(["Test input"])
        
        assert str(exc_info.value) == "Model error"

    def test_text_model_empty_input(self, text_model):
        """Test TextModel with empty input."""
        predictions = text_model.predict([])
        assert predictions == []

    def test_text_model_invalid_input(self, text_model):
        """Test TextModel with invalid input."""
        with pytest.raises(TypeError):
            text_model.predict(None)

    def test_text_model_custom_parameters(self, text_model, mock_model):
        """Test TextModel with custom generation parameters."""
        inputs = ["Test input"]
        text_model.predict(inputs, temperature=0.8, max_tokens=100)
        
        mock_model.generate.assert_called_once()
        call_args = mock_model.generate.call_args[1]
        assert call_args["temperature"] == 0.8
        assert call_args["max_tokens"] == 100 