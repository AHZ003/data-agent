---
name: suggested_questions
version: 1
description: Initial suggested-questions seed prompt extracted from config.py.
---
Given this dataset profile, suggest 5 interesting and diverse
questions a business analyst might ask. Cover different analysis types:
1. A descriptive/summary question
2. A trend/time-based question
3. A comparison question
4. A prediction question
5. An anomaly/outlier question

Return ONLY a JSON array of 5 question strings. Example:
["What are the top 10 products by revenue?", "How has monthly sales trended over time?", ...]
