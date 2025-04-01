"""
Progress Bar Module

This module provides classes for visualizing progress in the console:
- ProgressBar: Display a single progress bar
- MultiProgressBarManager: Manage multiple progress bars
"""

import os
import sys
import time
import threading
from typing import Dict, Optional, List, Union

# Import our progress tracker
from .progress_tracker import ProgressTracker


class ProgressBar:
    """
    Console-based progress bar for visualizing task progress.
    
    Displays a graphical progress bar in the console, showing completion
    percentage, task description, and time estimates.
    """
    
    def __init__(
        self,
        tracker: ProgressTracker,
        width: int = 40,
        fill_char: str = "█",
        empty_char: str = "░",
        refresh_rate: float = 0.1,
        clear_on_complete: bool = False
    ):
        """
        Initialize a new progress bar linked to a tracker.
        
        Args:
            tracker: ProgressTracker to visualize
            width: Width of the progress bar in characters
            fill_char: Character used for filled portion of the bar
            empty_char: Character used for empty portion of the bar
            refresh_rate: How often to refresh the display (seconds)
            clear_on_complete: Whether to clear the bar when complete
        """
        self.tracker = tracker
        self.width = max(10, width)
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.refresh_rate = refresh_rate
        self.clear_on_complete = clear_on_complete
        
        # Register callbacks with the tracker
        tracker.add_callback("update", lambda t: self.render())
        if clear_on_complete:
            tracker.add_callback("complete", lambda t: self.clear())
        
        # For controlling continuous display
        self._stop_display = threading.Event()
        self._display_thread: Optional[threading.Thread] = None
    
    def render(self) -> None:
        """Render the progress bar to the console."""
        # Calculate the filled width based on progress percentage
        progress = self.tracker.percentage / 100.0
        filled_width = int(self.width * progress)
        
        # Create the bar with filled and empty portions
        bar = self.fill_char * filled_width + self.empty_char * (self.width - filled_width)
        
        # Get the percentage and description
        percent = f"{self.tracker.percentage:.0f}%"
        description = self.tracker.description
        
        # Get time information
        elapsed = self.tracker.format_time(self.tracker.elapsed_time)
        eta = "" if self.tracker.eta is None else f"ETA: {self.tracker.format_time(self.tracker.eta)}"
        
        # Create the full progress line
        progress_line = f"|{bar}| {percent} | {description} | {elapsed}"
        if eta:
            progress_line += f" {eta}"
        
        # Add status message if available
        if self.tracker.status_message:
            status_snippet = self.tracker.status_message
            # Truncate if too long
            max_status_len = max(10, os.get_terminal_size().columns - len(progress_line) - 10)
            if len(status_snippet) > max_status_len:
                status_snippet = status_snippet[:max_status_len-3] + "..."
            progress_line += f" | {status_snippet}"
        
        # Print the progress bar, overwriting the current line
        sys.stdout.write("\r" + progress_line)
        sys.stdout.flush()
        
        # Add newline when complete
        if self.tracker.completed:
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    def clear(self) -> None:
        """Clear the progress bar from the console."""
        sys.stdout.write("\r" + " " * os.get_terminal_size().columns)
        sys.stdout.write("\r")
        sys.stdout.flush()
    
    def start_continuous_display(self) -> None:
        """
        Start a thread that continuously updates the progress bar.
        This is useful when progress updates happen in another thread.
        """
        if self._display_thread is not None and self._display_thread.is_alive():
            return  # Already running
        
        self._stop_display.clear()
        
        def display_loop():
            """Continuously update the progress bar until stopped."""
            while not self._stop_display.is_set():
                self.render()
                time.sleep(self.refresh_rate)
                if self.tracker.completed:
                    break
        
        self._display_thread = threading.Thread(target=display_loop)
        self._display_thread.daemon = True
        self._display_thread.start()
    
    def stop_continuous_display(self) -> None:
        """Stop the continuous display thread."""
        if self._display_thread is not None:
            self._stop_display.set()
            self._display_thread.join(timeout=1.0)
            self._display_thread = None


class MultiProgressBarManager:
    """
    Manage and display multiple progress bars simultaneously.
    
    Handles the display of multiple progress bars in the console,
    ensuring they don't interfere with each other.
    """
    
    def __init__(self):
        """Initialize the progress bar manager."""
        self.bars: Dict[str, ProgressBar] = {}
        self.active = False
        self._stop_display = threading.Event()
        self._display_thread: Optional[threading.Thread] = None
        self.refresh_rate = 0.1
    
    def add_bar(self, bar_id: str, bar: ProgressBar) -> None:
        """
        Add a progress bar to manage.
        
        Args:
            bar_id: Unique identifier for the bar
            bar: ProgressBar instance to manage
        """
        self.bars[bar_id] = bar
    
    def remove_bar(self, bar_id: str) -> None:
        """
        Remove a progress bar from management.
        
        Args:
            bar_id: Identifier of the bar to remove
        """
        if bar_id in self.bars:
            del self.bars[bar_id]
    
    def start(self) -> None:
        """Start displaying the managed progress bars."""
        if self.active:
            return
        
        self.active = True
        self._stop_display.clear()
        
        def display_loop():
            """Continuously update all progress bars."""
            while not self._stop_display.is_set():
                self.update_display()
                time.sleep(self.refresh_rate)
                
                # Check if all bars are complete
                if all(bar.tracker.completed for bar in self.bars.values()):
                    # Add a final newline after all bars
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    break
        
        self._display_thread = threading.Thread(target=display_loop)
        self._display_thread.daemon = True
        self._display_thread.start()
    
    def stop(self) -> None:
        """Stop displaying the progress bars."""
        if not self.active:
            return
        
        self._stop_display.set()
        if self._display_thread:
            self._display_thread.join(timeout=1.0)
        
        # Clear the display
        num_bars = len(self.bars)
        sys.stdout.write("\033[%dA" % num_bars)  # Move cursor up N lines
        for _ in range(num_bars):
            sys.stdout.write("\r" + " " * os.get_terminal_size().columns)
            sys.stdout.write("\n")
        
        sys.stdout.write("\r")
        sys.stdout.flush()
        
        self.active = False
        self._display_thread = None
    
    def update_display(self) -> None:
        """Update the display of all progress bars."""
        if not self.bars:
            return
        
        # Get terminal width
        terminal_width = os.get_terminal_size().columns
        
        # Clear the display area first
        num_bars = len(self.bars)
        sys.stdout.write("\033[%dA" % num_bars)  # Move cursor up N lines
        
        # Render each bar
        for bar_id, bar in self.bars.items():
            # Calculate the filled width based on progress percentage
            progress = bar.tracker.percentage / 100.0
            filled_width = int(bar.width * progress)
            
            # Create the bar with filled and empty portions
            bar_str = bar.fill_char * filled_width + bar.empty_char * (bar.width - filled_width)
            
            # Get the percentage and description
            percent = f"{bar.tracker.percentage:.0f}%"
            description = bar.tracker.description[:20]  # Limit length
            
            # Create the full progress line
            progress_line = f"|{bar_str}| {percent} | {description}"
            
            # Add time info if there's room
            if terminal_width > len(progress_line) + 10:
                elapsed = bar.tracker.format_time(bar.tracker.elapsed_time)
                progress_line += f" | {elapsed}"
                
                if bar.tracker.eta is not None and not bar.tracker.completed:
                    eta = bar.tracker.format_time(bar.tracker.eta)
                    if terminal_width > len(progress_line) + 10:
                        progress_line += f" ETA: {eta}"
            
            # Print the progress bar, padded to terminal width
            sys.stdout.write("\r" + progress_line.ljust(terminal_width))
            sys.stdout.write("\n")
        
        sys.stdout.flush() 