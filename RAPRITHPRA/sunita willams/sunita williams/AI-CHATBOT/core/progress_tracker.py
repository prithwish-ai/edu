"""
Progress Tracker Module

This module provides classes for tracking and reporting progress in time-consuming operations.
It includes:
- ProgressTracker: Track progress of a single task
- MultiProgressTracker: Manage progress for multiple related tasks
"""

import time
from typing import Dict, List, Optional, Any, Callable, Union


class ProgressTracker:
    """
    Track progress for a single task.
    
    Tracks current progress, elapsed time, and estimates completion time
    for a task with a known number of steps.
    """
    
    def __init__(
        self,
        total_steps: int,
        description: str = "Task",
    ):
        """
        Initialize a new progress tracker.
        
        Args:
            total_steps: Total number of steps in the task
            description: Description of the task
        
        Raises:
            ValueError: If total_steps is less than 1
        """
        if total_steps < 1:
            raise ValueError("Total steps must be at least 1")
        
        self.total_steps = total_steps
        self.current_steps = 0
        self.description = description
        self.status_message = ""
        self.completed = False
        self.start_time = time.time()
        self._callbacks = {"update": [], "complete": []}
    
    def update(self, steps: int = 1, message: Optional[str] = None) -> None:
        """
        Update progress by a specified number of steps.
        
        Args:
            steps: Number of steps completed
            message: Optional status message
        """
        if self.completed:
            return
        
        # Update step count
        self.current_steps += steps
        if self.current_steps >= self.total_steps:
            self.current_steps = self.total_steps
            self.completed = True
        
        # Update status message if provided
        if message is not None:
            self.status_message = message
        
        # Execute update callbacks
        for callback in self._callbacks["update"]:
            callback(self)
        
        # Execute completion callbacks if task is now complete
        if self.completed:
            for callback in self._callbacks["complete"]:
                callback(self)
    
    def add_callback(self, event: str, callback: Callable[["ProgressTracker"], None]) -> None:
        """
        Add a callback function that is called when progress is updated or completed.
        
        Args:
            event: Event to trigger callback ("update" or "complete")
            callback: Function to call with the tracker as argument
            
        Raises:
            ValueError: If event is not recognized
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}. Use 'update' or 'complete'")
        
        self._callbacks[event].append(callback)
    
    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        self.current_steps = 0
        self.status_message = ""
        self.completed = False
        self.start_time = time.time()
    
    @property
    def percentage(self) -> float:
        """Calculate the percentage of completion."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_steps / self.total_steps) * 100.0
    
    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time in seconds since tracking started."""
        return time.time() - self.start_time
    
    @property
    def eta(self) -> Optional[float]:
        """
        Estimate time remaining in seconds.
        
        Returns None if no steps have been completed or if task is complete.
        """
        if self.current_steps == 0 or self.completed:
            return 0 if self.completed else None
        
        # Calculate time per step and multiply by remaining steps
        time_per_step = self.elapsed_time / self.current_steps
        return time_per_step * (self.total_steps - self.current_steps)
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in a human-readable format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (e.g., "5.2s", "1.5m", "2.0h")
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def __str__(self) -> str:
        """Create a string representation of current progress."""
        progress = f"{self.description}: {self.current_steps}/{self.total_steps} | {self.percentage:.1f}%"
        
        # Add elapsed time
        progress += f" | Elapsed: {self.format_time(self.elapsed_time)}"
        
        # Add ETA if available
        if self.eta is not None and not self.completed:
            progress += f" | ETA: {self.format_time(self.eta)}"
        
        # Add status message if available
        if self.status_message:
            progress += f" | {self.status_message}"
        
        return progress


class MultiProgressTracker:
    """
    Track progress for multiple related tasks.
    
    Manages multiple ProgressTracker instances and provides
    aggregated progress information.
    """
    
    def __init__(self, description: str = "Multi-task"):
        """
        Initialize a new multi-progress tracker.
        
        Args:
            description: Description of the overall task
        """
        self.description = description
        self.trackers: Dict[str, ProgressTracker] = {}
    
    def add_tracker(
        self,
        task_id: str,
        total_steps: int,
        description: str = None
    ) -> ProgressTracker:
        """
        Add a new tracker for a task.
        
        Args:
            task_id: Unique identifier for the task
            total_steps: Total number of steps for the task
            description: Description of the task (defaults to task_id if None)
            
        Returns:
            The newly created tracker
            
        Raises:
            ValueError: If task_id already exists
        """
        # Use task_id as description if none provided
        if description is None:
            description = task_id
        
        # Create and store the tracker
        tracker = ProgressTracker(total_steps, description)
        self.trackers[task_id] = tracker
        
        return tracker
    
    def update(
        self,
        task_id: str,
        steps: int = 1,
        message: Optional[str] = None
    ) -> None:
        """
        Update progress for a specific task.
        
        Args:
            task_id: Task identifier
            steps: Number of steps completed
            message: Optional status message
            
        Raises:
            KeyError: If task_id does not exist
        """
        if task_id not in self.trackers:
            raise KeyError(f"Task '{task_id}' does not exist")
        
        self.trackers[task_id].update(steps, message)
    
    def get_overall_percentage(self) -> float:
        """
        Calculate the overall percentage across all tasks.
        
        Returns the average percentage of all tasks.
        """
        if not self.trackers:
            return 0.0
        
        total_percentage = sum(tracker.percentage for tracker in self.trackers.values())
        return total_percentage / len(self.trackers)
    
    def get_total_elapsed_time(self) -> float:
        """
        Get the total elapsed time across all tasks.
        
        Returns the maximum elapsed time among all tasks.
        """
        if not self.trackers:
            return 0.0
        
        return max(tracker.elapsed_time for tracker in self.trackers.values())
    
    def get_summary(self) -> str:
        """
        Get a summary of all tasks and their progress.
        
        Returns a multi-line string with overall progress and individual task progress.
        """
        if not self.trackers:
            return f"{self.description} - No tasks registered"
        
        # Create summary string
        overall = self.get_overall_percentage()
        summary = f"{self.description} - Overall: {overall:.1f}%\n"
        
        # Add each task's progress
        for task_id, tracker in self.trackers.items():
            summary += f"  - {tracker}\n"
        
        return summary.rstrip()
