import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.serving.app import app
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    return TestClient(app)
