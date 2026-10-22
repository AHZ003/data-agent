"""VLM-based image-to-table extraction using Gemini Vision."""

from google import genai
from google.genai import types as genai_types
from PIL import Image
import io
from typing import Optional, Tuple

from config import MODEL_NAME, VISION_EXTRACTION_PROMPT
from core import tracing
from core import llm


def extract_table_from_image(
    image_bytes: bytes, mime_type: str = "image/png"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract table data from an image using Gemini Vision.

    Returns: (csv_string, error_message)
    """
    client = llm.client()

    try:
        image = Image.open(io.BytesIO(image_bytes))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[VISION_EXTRACTION_PROMPT, image],
            config=genai_types.GenerateContentConfig(
                max_output_tokens=llm.OUTPUT_BUDGET,
            ),
        )
        tracing.record_usage(response, MODEL_NAME)
        result = response.text.strip()
        if result == "NOT_A_TABLE":
            return None, "The uploaded image does not contain a recognizable table."
        return result, None
    except Exception as e:
        return None, f"Vision extraction failed: {str(e)}"
