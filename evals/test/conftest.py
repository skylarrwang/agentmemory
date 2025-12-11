"""Shared fixtures for evaluation tests"""

import pytest
import tempfile
import shutil
from pathlib import Path
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def short_term_memory(temp_dir, monkeypatch):
    """Create a ShortTermMemory instance with temp storage"""
    # Change working directory to temp dir
    monkeypatch.chdir(temp_dir)
    return ShortTermMemory("test_user", session_num=1)

@pytest.fixture
def long_term_memory(temp_dir, monkeypatch):
    """Create a LongTermMemory instance with temp storage"""
    monkeypatch.chdir(temp_dir)
    return LongTermMemory("test_user")

