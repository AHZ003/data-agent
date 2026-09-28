# Error taxonomy

How failures are labeled for error analysis. Sample 50 misses per benchmark with
`python -m benchmarks.error_sample sample ...`, label each row by hand with one
primary category, then run `summarize`. Labels live in `docs/error_analysis/`.

Pick the **first** thing that went wrong: a query that links the wrong column and
therefore also aggregates wrongly is `schema_linking`.

| Category | Use when | Example |
|---|---|---|
| `schema_linking` | The wrong table or column was used, or a needed one is missing | Uses `Track.UnitPrice` when the question is about what customers paid (`InvoiceLine.UnitPrice`) |
| `join` | Wrong join path, missing or extra join, wrong key | Joins `Customer` to `Employee` on `CustomerId = EmployeeId` |
| `aggregation` | Wrong aggregate, GROUP BY, HAVING or DISTINCT | `COUNT(*)` where `COUNT(DISTINCT CustomerId)` was asked |
| `value_mismatch` | A literal doesn't match how the value is stored | `WHERE state = 'California'` but the column stores `'CA'` |
| `condition` | Wrong filter logic, comparison or NULL handling | `>` instead of `>=`; `= NULL` |
| `ordering_limit` | Wrong sort direction, LIMIT, or tie handling | Ascending when the question asks for the largest |
| `output_shape` | Logic right, but columns extra/missing/wrong | Returns `id, name` when only `name` was asked (scored as a miss) |
| `ambiguous_question` | The question genuinely allows the predicted reading | "best customer" read as most orders instead of most revenue |
| `questionable_gold` | The gold SQL is wrong or arguably wrong | Gold misses a DISTINCT the question implies |
| `execution_error` | No executable SQL after self-repair | Repeated syntax or unknown-column errors |
| `other` | None of the above; explain in `notes` | |

`ambiguous_question` and `questionable_gold` matter for honest reporting: they
bound how much of the gap to 100% is recoverable at all.
