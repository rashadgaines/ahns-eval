"""Results management system for the evaluation framework."""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from eval_framework.core.base import EvalResult


class ResultMetadata(BaseModel):
    """Metadata for evaluation results.
    
    Attributes:
        run_id: Unique identifier for the evaluation run
        timestamp: When the evaluation was run
        model_name: Name of the evaluated model
        dataset_name: Name of the dataset used
        config: Configuration used for the evaluation
        system_info: System information where evaluation was run
        git_info: Git repository information if available
    """

    run_id: UUID = Field(default_factory=uuid4, description="Unique run identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Evaluation timestamp"
    )
    model_name: str = Field(..., description="Name of the evaluated model")
    dataset_name: str = Field(..., description="Name of the dataset used")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluation configuration"
    )
    system_info: Dict[str, Any] = Field(
        default_factory=dict, description="System information"
    )
    git_info: Optional[Dict[str, Any]] = Field(
        None, description="Git repository information"
    )


class ResultStore(ABC):
    """Abstract base class for result storage backends."""

    @abstractmethod
    def save(self, result: EvalResult, metadata: ResultMetadata) -> None:
        """Save evaluation results.
        
        Args:
            result: The evaluation results to save
            metadata: Metadata about the evaluation run
        """
        pass

    @abstractmethod
    def load(self, run_id: UUID) -> tuple[EvalResult, ResultMetadata]:
        """Load evaluation results.
        
        Args:
            run_id: ID of the evaluation run to load
            
        Returns:
            Tuple of (results, metadata)
            
        Raises:
            KeyError: If the run_id is not found
        """
        pass

    @abstractmethod
    def list_runs(
        self, model_name: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> List[UUID]:
        """List available evaluation runs.
        
        Args:
            model_name: Optional filter by model name
            dataset_name: Optional filter by dataset name
            
        Returns:
            List of run IDs
        """
        pass


class FileResultStore(ResultStore):
    """File-based result storage backend.
    
    This backend stores results in JSON files in a directory structure.
    """

    def __init__(self, base_dir: Union[str, Path]):
        """Initialize the file store.
        
        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, run_id: UUID) -> Path:
        """Get the directory for a specific run.
        
        Args:
            run_id: The run ID
            
        Returns:
            Path to the run directory
        """
        return self.base_dir / str(run_id)

    def save(self, result: EvalResult, metadata: ResultMetadata) -> None:
        """Save results to files.
        
        Args:
            result: The evaluation results to save
            metadata: Metadata about the evaluation run
        """
        run_dir = self._get_run_dir(metadata.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(run_dir / "results.json", "w") as f:
            f.write(result.model_dump_json(indent=2))

        # Save metadata
        with open(run_dir / "metadata.json", "w") as f:
            f.write(metadata.model_dump_json(indent=2))

    def load(self, run_id: UUID) -> tuple[EvalResult, ResultMetadata]:
        """Load results from files.
        
        Args:
            run_id: ID of the evaluation run to load
            
        Returns:
            Tuple of (results, metadata)
            
        Raises:
            KeyError: If the run_id is not found
        """
        run_dir = self._get_run_dir(run_id)
        if not run_dir.exists():
            raise KeyError(f"Run not found: {run_id}")

        # Load results
        with open(run_dir / "results.json") as f:
            result = EvalResult.model_validate_json(f.read())

        # Load metadata
        with open(run_dir / "metadata.json") as f:
            metadata = ResultMetadata.model_validate_json(f.read())

        return result, metadata

    def list_runs(
        self, model_name: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> List[UUID]:
        """List available runs.
        
        Args:
            model_name: Optional filter by model name
            dataset_name: Optional filter by dataset name
            
        Returns:
            List of run IDs
        """
        run_ids = []
        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue

            try:
                run_id = UUID(run_dir.name)
                _, metadata = self.load(run_id)

                if model_name and metadata.model_name != model_name:
                    continue
                if dataset_name and metadata.dataset_name != dataset_name:
                    continue

                run_ids.append(run_id)
            except (ValueError, KeyError):
                continue

        return run_ids


class SQLiteResultStore(ResultStore):
    """SQLite-based result storage backend."""

    def __init__(self, db_path: Union[str, Path]):
        """Initialize the SQLite store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    system_info TEXT NOT NULL,
                    git_info TEXT,
                    results TEXT NOT NULL
                )
            """)

    def save(self, result: EvalResult, metadata: ResultMetadata) -> None:
        """Save results to SQLite.
        
        Args:
            result: The evaluation results to save
            metadata: Metadata about the evaluation run
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, timestamp, model_name, dataset_name,
                    config, system_info, git_info, results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(metadata.run_id),
                    metadata.timestamp.isoformat(),
                    metadata.model_name,
                    metadata.dataset_name,
                    json.dumps(metadata.config),
                    json.dumps(metadata.system_info),
                    json.dumps(metadata.git_info) if metadata.git_info else None,
                    result.model_dump_json(),
                ),
            )

    def load(self, run_id: UUID) -> tuple[EvalResult, ResultMetadata]:
        """Load results from SQLite.
        
        Args:
            run_id: ID of the evaluation run to load
            
        Returns:
            Tuple of (results, metadata)
            
        Raises:
            KeyError: If the run_id is not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (str(run_id),)
            )
            row = cursor.fetchone()
            if not row:
                raise KeyError(f"Run not found: {run_id}")

            metadata = ResultMetadata(
                run_id=UUID(row[0]),
                timestamp=datetime.fromisoformat(row[1]),
                model_name=row[2],
                dataset_name=row[3],
                config=json.loads(row[4]),
                system_info=json.loads(row[5]),
                git_info=json.loads(row[6]) if row[6] else None,
            )

            result = EvalResult.model_validate_json(row[7])

            return result, metadata

    def list_runs(
        self, model_name: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> List[UUID]:
        """List available runs.
        
        Args:
            model_name: Optional filter by model name
            dataset_name: Optional filter by dataset name
            
        Returns:
            List of run IDs
        """
        query = "SELECT run_id FROM runs"
        params = []
        conditions = []

        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if dataset_name:
            conditions.append("dataset_name = ?")
            params.append(dataset_name)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [UUID(row[0]) for row in cursor.fetchall()]


class ResultManager:
    """Manager for evaluation results.
    
    This class provides a high-level interface for managing evaluation results,
    including saving, loading, and querying results.
    """

    def __init__(self, store: ResultStore):
        """Initialize the result manager.
        
        Args:
            store: The result storage backend to use
        """
        self.store = store

    def save_results(
        self,
        result: EvalResult,
        model_name: str,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None,
        system_info: Optional[Dict[str, Any]] = None,
        git_info: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Save evaluation results.
        
        Args:
            result: The evaluation results to save
            model_name: Name of the evaluated model
            dataset_name: Name of the dataset used
            config: Optional evaluation configuration
            system_info: Optional system information
            git_info: Optional git repository information
            
        Returns:
            The run ID of the saved results
        """
        metadata = ResultMetadata(
            model_name=model_name,
            dataset_name=dataset_name,
            config=config or {},
            system_info=system_info or {},
            git_info=git_info,
        )

        self.store.save(result, metadata)
        return metadata.run_id

    def load_results(self, run_id: UUID) -> tuple[EvalResult, ResultMetadata]:
        """Load evaluation results.
        
        Args:
            run_id: ID of the evaluation run to load
            
        Returns:
            Tuple of (results, metadata)
        """
        return self.store.load(run_id)

    def list_runs(
        self, model_name: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> List[UUID]:
        """List available evaluation runs.
        
        Args:
            model_name: Optional filter by model name
            dataset_name: Optional filter by dataset name
            
        Returns:
            List of run IDs
        """
        return self.store.list_runs(model_name, dataset_name)

    def get_results_dataframe(
        self, model_name: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Get results as a pandas DataFrame.
        
        Args:
            model_name: Optional filter by model name
            dataset_name: Optional filter by dataset name
            
        Returns:
            DataFrame containing the results
        """
        run_ids = self.list_runs(model_name, dataset_name)
        rows = []

        for run_id in run_ids:
            result, metadata = self.load_results(run_id)
            row = {
                "run_id": str(run_id),
                "timestamp": metadata.timestamp,
                "model_name": metadata.model_name,
                "dataset_name": metadata.dataset_name,
                **result.metrics,
            }
            rows.append(row)

        return pd.DataFrame(rows) 