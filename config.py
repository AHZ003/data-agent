"""Configuration and constants for DataAgent."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- Model Configuration ---
# Override via DATAAGENT_MODEL env var for cross-model eval comparisons.
MODEL_NAME = os.getenv("DATAAGENT_MODEL", "gemini-flash-latest")
MAX_TOKENS = 4096
TEMPERATURE = 0.0

# --- Database ---
SQLITE_DB_NAME = ":memory:"
DEFAULT_TABLE_NAME = "uploaded_data"
MAX_QUERY_ROWS = 1000

# --- Agent Configuration ---
MAX_RETRY_ATTEMPTS = 3
CRITIC_CONFIDENCE_THRESHOLD_WARN = 70
CRITIC_CONFIDENCE_THRESHOLD_REJECT = 40

# --- RAG ---
CHROMA_COLLECTION_NAME = "statistical_knowledge"
CHROMA_PERSIST_DIR = "./rag/chroma_db"

# --- File Upload ---
ALLOWED_FILE_TYPES = ["csv", "xlsx", "xls", "png", "jpg", "jpeg"]
MAX_DISPLAY_ROWS = 10

# --- Prompts ---
# Prompts are versioned in `prompts/*.md` with YAML frontmatter.
# See prompts/__init__.py for the registry and loader. Constants below
# are kept for backward-compat with existing `from config import X`
# callers — they resolve to the file body at import time.
from prompts import get as _prompt  # noqa: E402

SCHEMA_AGENT_SYSTEM_PROMPT = _prompt("schema_agent")
CODER_AGENT_SYSTEM_PROMPT = _prompt("coder_agent")
PLANNER_AGENT_SYSTEM_PROMPT = _prompt("planner_agent")
CRITIC_AGENT_SYSTEM_PROMPT = _prompt("critic_agent")
STORYTELLER_AGENT_SYSTEM_PROMPT = _prompt("storyteller_agent")
SUGGESTED_QUESTIONS_PROMPT = _prompt("suggested_questions")
VISION_EXTRACTION_PROMPT = _prompt("vision_extraction")
