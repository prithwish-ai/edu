import time
from typing import Optional, Dict, Any, List, Callable


class ProgressTracker:
    """
    A utility class to track progress for tasks and processes.
    Provides functionality to update, display, and estimate completion time.
    """
    
    def __init__(self, total_steps: int, description: str = "Task", show_percentage: bool = True, 
                 show_elapsed_time: bool = True, show_eta: bool = True):
        """
        Initialize a new progress tracker.
        
        Args:
            total_steps (int): Total number of steps to complete the task
            description (str): Description of the task being tracked
            show_percentage (bool): Whether to show percentage in progress string
            show_elapsed_time (bool): Whether to show elapsed time
            show_eta (bool): Whether to show estimated time to completion
        """
        if total_steps <= 0:
            raise ValueError("Total steps must be greater than zero")
            
        self.total_steps = total_steps
        self.current_steps = 0
        self.description = description
        self.show_percentage = show_percentage
        self.show_elapsed_time = show_elapsed_time
        self.show_eta = show_eta
        
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.completed = False
        self.step_history: List[Dict[str, Any]] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "update": [],
            "complete": []
        }
    
    def update(self, steps: int = 1, message: Optional[str] = None) -> None:
        """
        Update progress by the specified number of steps.
        
        Args:
            steps (int): Number of steps to increment by
            message (str, optional): Optional message to associate with this update
        """
        if self.completed:
            return
            
        self.current_steps += steps
        current_time = time.time()
        
        # Record step information
        step_info = {
            "steps_added": steps,
            "current_steps": self.current_steps,
            "time": current_time,
            "time_since_last": current_time - self.last_update_time,
            "message": message
        }
        self.step_history.append(step_info)
        self.last_update_time = current_time
        
        # Check if completed
        if self.current_steps >= self.total_steps:
            self.current_steps = self.total_steps
            self.completed = True
            self._trigger_callbacks("complete")
            
        # Trigger update callbacks
        self._trigger_callbacks("update")
    
    def reset(self) -> None:
        """Reset the progress tracker to its initial state."""
        self.current_steps = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.completed = False
        self.step_history = []
    
    def add_callback(self, event_type: str, callback: Callable) -> None:
        """
        Add a callback to be triggered on specific events.
        
        Args:
            event_type (str): Event type ('update' or 'complete')
            callback (Callable): Function to call when event occurs
        """
        if event_type not in self._callbacks:
            raise ValueError(f"Unknown event type: {event_type}")
        self._callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str) -> None:
        """Trigger all callbacks for the given event type."""
        for callback in self._callbacks.get(event_type, []):
            callback(self)
    
    @property
    def percentage(self) -> float:
        """Get the current progress as a percentage."""
        if self.total_steps == 0:
            return 100.0
        return (self.current_steps / self.total_steps) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """
        Estimate the remaining time based on the current progress and elapsed time.
        
        Returns:
            Optional[float]: Estimated time remaining in seconds, or None if no steps completed
        """
        if self.current_steps == 0 or self.completed:
            return None
            
        elapsed = self.elapsed_time
        progress_ratio = self.current_steps / self.total_steps
        total_estimated_time = elapsed / progress_ratio
        
        return max(0, total_estimated_time - elapsed)
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a human-readable string.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_progress_string(self) -> str:
        """
        Get a formatted string representing the current progress.
        
        Returns:
            str: Formatted progress string
        """
        parts = [f"{self.description}: {self.current_steps}/{self.total_steps}"]
        
        if self.show_percentage:
            parts.append(f"{self.percentage:.1f}%")
            
        if self.show_elapsed_time:
            parts.append(f"Elapsed: {self.format_time(self.elapsed_time)}")
            
        if self.show_eta and not self.completed and self.estimated_time_remaining is not None:
            parts.append(f"ETA: {self.format_time(self.estimated_time_remaining)}")
            
        if self.completed:
            parts.append("(Completed)")
            
        return " | ".join(parts)
    
    def __str__(self) -> str:
        """String representation of the progress tracker."""
        return self.get_progress_string()


class MultiProgressTracker:
    """
    Track progress for multiple tasks simultaneously.
    Useful for tracking parallel or sequential subtasks.
    """
    
    def __init__(self, description: str = "Multi-task Progress"):
        """
        Initialize a new multi-progress tracker.
        
        Args:
            description (str): Overall description for the set of tasks
        """
        self.description = description
        self.trackers: Dict[str, ProgressTracker] = {}
        self.start_time = time.time()
    
    def add_tracker(self, name: str, total_steps: int, description: Optional[str] = None) -> ProgressTracker:
        """
        Add a new progress tracker for a subtask.
        
        Args:
            name (str): Unique identifier for this tracker
            total_steps (int): Total number of steps for this subtask
            description (str, optional): Description for this subtask
            
        Returns:
            ProgressTracker: The newly created tracker
        """
        if name in self.trackers:
            raise ValueError(f"Tracker with name '{name}' already exists")
            
        tracker_description = description or name
        tracker = ProgressTracker(total_steps, tracker_description)
        self.trackers[name] = tracker
        return tracker
    
    def get_tracker(self, name: str) -> ProgressTracker:
        """
        Get a specific tracker by name.
        
        Args:
            name (str): Name of the tracker to retrieve
            
        Returns:
            ProgressTracker: The requested tracker
        """
        if name not in self.trackers:
            raise ValueError(f"No tracker found with name '{name}'")
        return self.trackers[name]
    
    def update(self, name: str, steps: int = 1, message: Optional[str] = None) -> None:
        """
        Update a specific tracker by name.
        
        Args:
            name (str): Name of the tracker to update
            steps (int): Number of steps to increment
            message (str, optional): Optional message to associate with this update
        """
        self.get_tracker(name).update(steps, message)
    
    @property
    def overall_progress(self) -> float:
        """
        Calculate the average progress across all trackers.
        
        Returns:
            float: Overall progress as a percentage
        """
        if not self.trackers:
            return 0.0
            
        total_percentage = sum(tracker.percentage for tracker in self.trackers.values())
        return total_percentage / len(self.trackers)
    
    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time in seconds since the multi-tracker was created."""
        return time.time() - self.start_time
    
    def is_complete(self) -> bool:
        """Check if all trackers are complete."""
        if not self.trackers:
            return False
        return all(tracker.completed for tracker in self.trackers.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked tasks.
        
        Returns:
            Dict: Summary information
        """
        return {
            "description": self.description,
            "overall_progress": self.overall_progress,
            "elapsed_time": self.elapsed_time,
            "complete": self.is_complete(),
            "tasks": {
                name: {
                    "description": tracker.description,
                    "progress": tracker.percentage,
                    "steps": f"{tracker.current_steps}/{tracker.total_steps}",
                    "complete": tracker.completed
                }
                for name, tracker in self.trackers.items()
            }
        }
    
    def __str__(self) -> str:
        """String representation of the multi-progress tracker."""
        parts = [f"{self.description}: {self.overall_progress:.1f}% complete"]
        
        for name, tracker in self.trackers.items():
            status = "✓" if tracker.completed else " "
            parts.append(f"  [{status}] {name}: {tracker.percentage:.1f}%")
            
        return "\n".join(parts) 