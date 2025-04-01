"""
Core Package for ITI Chatbot

This package contains the core functionality for the ITI Chatbot application,
including progress tracking, AI integration, and utility functions.
"""

from .progress_tracker import ProgressTracker, MultiProgressTracker
from .progress_bar import ProgressBar, MultiProgressBarManager

__all__ = [
    'ProgressTracker',
    'MultiProgressTracker',
    'ProgressBar',
    'MultiProgressBarManager',
]

__version__ = '0.1.0' 