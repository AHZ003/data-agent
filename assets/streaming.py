"""Streaming helpers to simulate progressive text output in Streamlit."""

import time
from typing import Generator


def type_stream(text: str, words_per_chunk: int = 2, delay: float = 0.02) -> Generator[str, None, None]:
    """Yield text in small word chunks with a delay — for st.write_stream."""
    if not text:
        return
    words = text.split(" ")
    buf = []
    for i, w in enumerate(words):
        buf.append(w)
        if len(buf) >= words_per_chunk or i == len(words) - 1:
            yield " ".join(buf) + (" " if i < len(words) - 1 else "")
            buf = []
            time.sleep(delay)
