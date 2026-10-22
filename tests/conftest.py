"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_state():
    """Drei Knoten mit Grad 2, 3, 1, angekommen zu den Zeiten 0, 1, 2; nun trifft Knoten 3 ein.
    Handrechnung der Gewichte (a = 1, A = 0): 2, 3, 1; mit Alterung nu = 1 mal (3 − Ankunft + 1)^(−1) = 1/4, 1/3, 1/2: 0,5, 1, 0,5; mit Fitness 1, 0,5, 0,25 zusätzlich: 0,5, 0,5, 0,125."""
    return np.array([2, 3, 1]), np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.5, 0.25]), 3
