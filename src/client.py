"""
API Client Initialization

Clients:
- OpenAI: Uses OPENAI_API_KEY from environment (via python-dotenv)
- Google Vertex AI: Uses GOOGLE_APPLICATION_CREDENTIALS env var pointing to
  a service account JSON file. Optional -- returns None if not configured.
"""
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# OpenAI client - picks up OPENAI_API_KEY from environment automatically
client = AsyncOpenAI()

# Google client - lazily initialized
_google_client = None
_google_client_initialized = False


def get_google_client():
    """
    Get Google Vertex AI client. Returns None if not configured.

    Requires:
    - google-genai and google-auth packages installed
    - GOOGLE_APPLICATION_CREDENTIALS env var set to service account JSON path

    Optionally:
    - GOOGLE_PROJECT_ID env var (defaults to project from credentials file)
    - GOOGLE_LOCATION env var (defaults to "global")
    """
    global _google_client, _google_client_initialized

    if _google_client_initialized:
        return _google_client

    _google_client_initialized = True

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        print("Note: GOOGLE_APPLICATION_CREDENTIALS not set. Google Gemini unavailable.")
        return None

    try:
        import json
        from google.genai import Client, types
        from google.oauth2 import service_account

        with open(credentials_path, 'r') as f:
            service_account_info = json.load(f)

        credentials = service_account.Credentials.from_service_account_info(
            info=service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        project_id = os.getenv("GOOGLE_PROJECT_ID", service_account_info.get("project_id"))
        location = os.getenv("GOOGLE_LOCATION", "global")

        _google_client = Client(
            vertexai=True,
            project=project_id,
            location=location,
            credentials=credentials,
            http_options=types.HttpOptions(
                async_client_args={"ssl": False},
            )
        )

        print(f"Google Vertex AI client initialized (project: {project_id})")
        return _google_client

    except ImportError:
        print("Note: google-genai/google-auth not installed. Google Gemini unavailable.")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize Google client: {e}")
        return None
