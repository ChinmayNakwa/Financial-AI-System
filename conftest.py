import os

# Populate dummy credentials so `backend.config.Settings()` can be constructed
# during test collection without a real .env file. Tests never make live calls.
for _key in (
    "GOOGLE_API_KEY",
    "FRED_API_KEY",
    "TAVILY_API_KEY",
    "NEWS_API_KEY",
    "COINDESK_API_KEY",
    "POLYGON_API_KEY",
    "SEC_API_KEY",
):
    os.environ.setdefault(_key, "test-key")
