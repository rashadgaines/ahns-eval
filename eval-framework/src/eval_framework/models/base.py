"""Base model interface for all AI models in the evaluation framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class BaseModel(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all AI models.
    """

    @abstractmethod
    async def generate(self, input_data: InputT, **kwargs) -> OutputT:
        """
        Generate a single output from input data.
        Args:
            input_data: The input to the model.
            **kwargs: Additional generation parameters.
        Returns:
            The generated output.
        """
        pass

    @abstractmethod
    async def batch_generate(self, batch_inputs: List[InputT], **kwargs) -> List[OutputT]:
        """
        Generate outputs for a batch of inputs.
        Args:
            batch_inputs: List of input data.
            **kwargs: Additional generation parameters.
        Returns:
            List of generated outputs.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata/info about the model (e.g., name, version, parameters).
        Returns:
            Dictionary of model information.
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: InputT) -> Optional[str]:
        """
        Validate the input data for the model.
        Args:
            input_data: The input to validate.
        Returns:
            None if valid, or an error message string if invalid.
        """
        pass 