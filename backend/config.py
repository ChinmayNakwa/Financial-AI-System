# config.py

import os
from pathlib import Path  # Import the Path object
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# --- The Key Fix ---
# 1. Get the directory where this config.py file lives.
#    __file__ is a special variable that holds the path to the current script.
#    .parent gives us the directory containing the file.
#    In this case, it will be the absolute path to 'FINANCIAL-AI-SYSTEM/'.
this_directory = Path(__file__).parent

# 2. Construct the full, absolute path to the .env file.
dotenv_path = this_directory / ".env"

# 3. Load the .env file using its full path.
load_dotenv(dotenv_path=dotenv_path)
# --- End of Fix ---


class Settings(BaseSettings):
    # API Keys
    MISTRAL_API_KEY: str = Field(..., env="MISTRAL_API_KEY")
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")
    ALPHA_VANTAGE_API_KEY: str = Field(..., env="ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY: str = Field(..., env="FRED_API_KEY")
    TAVILY_API_KEY: str = Field(..., env="TAVILY_API_KEY")
    NEWS_API_KEY: str = Field(..., env="NEWS_API_KEY")
    COINDESK_API_KEY: str = Field(..., env="COINDESK_API_KEY")
    POLYGON_API_KEY: str = Field(..., env="POLYGON_API_KEY")
    SEC_API_KEY: str = Field(..., env="SEC_API_KEY")
    SEC_USER_AGENT: str = Field(
        default="YourName YourEmail@example.com",
        env="SEC_USER_AGENT"
    )

settings = Settings()