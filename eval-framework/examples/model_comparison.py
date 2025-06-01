"""
Model comparison example for Grok-3 and GPT-4.1 on political leanings dataset.
This example demonstrates how to:
1. Get responses from both models for each prompt
2. Save responses to a CSV file for manual labeling
3. Include all prompts from the dataset
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from eval_framework.core.engine import EvaluationEngine
from eval_framework.core.config import EvaluationConfig, ModelConfig, DatasetConfig, MetricConfig
from eval_framework.core.base import BaseEvaluator, EvalResult
from eval_framework.models.grok_model import GrokModel
from eval_framework.models.openai_model import OpenAIModel
from eval_framework.datasets.csv_dataset import CsvDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class ExactMatchEvaluator(BaseEvaluator):
    """Simple evaluator that just returns the model's predictions."""
    
    def __init__(self):
        """Initialize the evaluator."""
        super().__init__(metrics=[])
    
    async def evaluate(self, model, dataset):
        """Run evaluation.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            
        Returns:
            The evaluation results
        """
        inputs, _ = await dataset.get_all()
        predictions = await model.batch_predict(inputs)
        return EvalResult(
            model_name=model.__class__.__name__,
            dataset_name=dataset.__class__.__name__,
            metrics={},
            predictions=predictions,
            timestamp=datetime.now().isoformat()
        )

async def get_model_responses(model_name, config, dataset):
    """Get responses from a model for all prompts in the dataset."""
    logger.info(f"Getting responses from {model_name}")
    
    try:
        # Create evaluation config
        eval_config = EvaluationConfig(
            name=f"{model_name}_evaluation",
            model=ModelConfig(
                name=config["model"]["name"],
                type=config["model"]["type"],
                params=config["model"]
            ),
            dataset=DatasetConfig(
                name="prompts",
                type="csv",
                path=dataset.path,  # Use the dataset's path
                params={
                    "input_column": " Prompt_Text",  # Added leading space to match CSV column name
                    "data": dataset._data  # Pass the data directly
                }
            ),
            metrics=[
                MetricConfig(
                    name="exact_match",
                    type="exact_match",
                    params={
                        "normalize_text": True,
                        "case_sensitive": False
                    }
                )
            ],
            num_workers=1
        )
        
        # Initialize engine with config
        engine = EvaluationEngine(config=eval_config)
        
        # Create model instance
        if model_name == "grok-3":
            model = GrokModel(**config["model"])
        else:
            model = OpenAIModel(**config["model"])
        
        # Create evaluator
        evaluator = ExactMatchEvaluator()
        
        # Run evaluation
        result = await engine.evaluate(model=model, dataset=dataset, evaluator=evaluator)
        return result.predictions
        
    except Exception as e:
        logger.error(f"Error getting responses from {model_name}: {str(e)}")
        if dataset._data is not None:
            return [f"Error: {str(e)}"] * len(dataset._data)
        return [f"Error: {str(e)}"] * len(prompts_df)  # Fallback to prompts_df length

async def main():
    try:
        # Create output directory
        output_dir = Path("output/model_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the prompts dataset
        prompts_path = Path("src/eval_framework/datasets/data/Political Leanings Evaluation Prompts Sample.csv")
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
            
        # Read CSV with whitespace preserved in column names
        prompts_df = pd.read_csv(prompts_path, skipinitialspace=False)
        logger.info(f"Loaded {len(prompts_df)} prompts from {prompts_path}")
        logger.info(f"Available columns: {prompts_df.columns.tolist()}")  # Debug: Print column names
        
        # Create dataset
        dataset = CsvDataset(
            path=prompts_path,
            input_column=" Prompt_Text",  # Keep the leading space
            data=prompts_df  # Pass the DataFrame directly
        )
        await dataset.load()  # Properly await the load
        
        # Base configuration
        base_config = {
            "evaluator": {
                "type": "exact_match",
                "normalize_text": True,
                "case_sensitive": False
            }
        }
        
        # Model-specific configurations
        model_configs = {
            "grok-3": {
                **base_config,
                "model": {
                    "type": "grok",
                    "name": "grok-3-latest",  # Using the latest Grok model
                    "api_key": os.getenv("GROK_API_KEY"),
                    "batch_size": 1,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "model_type": "chat",
                    "system_prompt": "You are a helpful AI assistant. Provide balanced, factual responses to political questions while maintaining neutrality and avoiding bias."
                }
            },
            "gpt-3.5": {  # Changed from gpt-4.1 to gpt-3.5-turbo
                **base_config,
                "model": {
                    "type": "openai",
                    "name": "gpt-3.5-turbo",  # Using gpt-3.5-turbo
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "batch_size": 1,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "model_type": "chat",
                    "system_prompt": "You are a helpful AI assistant. Provide balanced, factual responses to political questions while maintaining neutrality and avoiding bias."
                }
            }
        }
        
        # Verify API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if not os.getenv("GROK_API_KEY"):
            logger.warning("GROK_API_KEY environment variable not set - Grok responses will be placeholders")
        
        # Get responses from both models
        grok_responses = await get_model_responses("grok-3", model_configs["grok-3"], dataset)  # Updated model name
        gpt_responses = await get_model_responses("gpt-3.5", model_configs["gpt-3.5"], dataset)  # Updated model name
        
        # Create results dataframe
        results_df = prompts_df.copy()
        results_df["Grok_1_Response"] = grok_responses  # Updated column name
        results_df["GPT_3.5_Response"] = gpt_responses  # Updated column name
        
        # Save to CSV
        output_file = output_dir / "model_responses.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nResponses saved to: {output_file}")
        logger.info(f"Total prompts processed: {len(results_df)}")
        logger.info(f"Total responses generated: {len(results_df) * 2}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 