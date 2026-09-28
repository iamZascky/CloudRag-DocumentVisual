import sys
import os

# Set root directory for cPanel Phusion Passenger
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# Set production environment variables
os.environ["AI_MODE"] = os.getenv("AI_MODE", "cloud")

from src.api.routes import app

# Passenger requires an ASGI-to-WSGI adapter or native ASGI callable named 'application'
try:
    from a2wsgi import ASGIMiddleware
    application = ASGIMiddleware(app)
except ImportError:
    # If a2wsgi is not yet installed, use native callable
    application = app
