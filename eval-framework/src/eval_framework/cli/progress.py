"""Progress tracking for evaluations."""

import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks evaluation progress with rich progress bars."""
    
    def __init__(
        self,
        total_steps: int,
        description: str = "Evaluation",
        checkpoint_dir: Optional[str] = None
    ):
        """Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps in evaluation
            description: Description of the evaluation
            checkpoint_dir: Directory to save checkpoints
        """
        self.total_steps = total_steps
        self.description = description
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.start_time = time.time()
        self.current_step = 0
        self.errors: List[Dict[str, Any]] = []
        
        # Initialize rich components
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        
        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self) -> None:
        """Start progress tracking."""
        self.task_id = self.progress.add_task(
            self.description,
            total=self.total_steps
        )
        self.progress.start()
    
    def update(
        self,
        step: int,
        description: Optional[str] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Update progress.
        
        Args:
            step: Current step number
            description: Optional description of current step
            error: Optional error that occurred
        """
        self.current_step = step
        
        # Update progress bar
        self.progress.update(
            self.task_id,
            completed=step,
            description=description or self.description
        )
        
        # Handle error
        if error:
            self.errors.append({
                "step": step,
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            })
            self.console.print(
                Panel(
                    f"[red]Error at step {step}:[/red]\n{str(error)}",
                    title="Error",
                    border_style="red"
                )
            )
        
        # Save checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint()
    
    def finish(self) -> None:
        """Finish progress tracking and display summary."""
        self.progress.stop()
        
        # Display summary
        table = Table(title="Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        elapsed_time = time.time() - self.start_time
        table.add_row("Total Steps", str(self.total_steps))
        table.add_row("Completed Steps", str(self.current_step))
        table.add_row("Elapsed Time", f"{elapsed_time:.2f}s")
        table.add_row("Errors", str(len(self.errors)))
        
        self.console.print(table)
        
        # Display errors if any
        if self.errors:
            error_table = Table(title="Errors")
            error_table.add_column("Step", style="cyan")
            error_table.add_column("Error", style="red")
            error_table.add_column("Time", style="yellow")
            
            for error in self.errors:
                error_table.add_row(
                    str(error["step"]),
                    error["error"],
                    error["timestamp"]
                )
            
            self.console.print(error_table)
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint to file."""
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "description": self.description,
            "start_time": self.start_time,
            "errors": self.errors,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str) -> Optional["ProgressTracker"]:
        """Load checkpoint from file.
        
        Args:
            checkpoint_dir: Directory containing checkpoint file
            
        Returns:
            ProgressTracker instance if checkpoint exists, None otherwise
        """
        checkpoint_file = Path(checkpoint_dir) / "checkpoint.json"
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        
        tracker = cls(
            total_steps=checkpoint["total_steps"],
            description=checkpoint["description"],
            checkpoint_dir=checkpoint_dir
        )
        
        tracker.current_step = checkpoint["current_step"]
        tracker.start_time = checkpoint["start_time"]
        tracker.errors = checkpoint["errors"]
        
        return tracker 