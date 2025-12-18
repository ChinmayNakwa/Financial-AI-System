import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.api import app

app = app