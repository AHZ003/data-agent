---
name: vision_extraction
version: 1
description: Initial image-to-CSV extraction prompt extracted from config.py.
---
Extract the data from this table image. Return it as CSV format
with headers. If the image is not a table, respond with exactly: NOT_A_TABLE

Rules:
- Preserve all column headers exactly as shown
- Preserve all numeric values without rounding
- Use commas as delimiters
- Include a header row
