"""The "Ask about your data" bar's question pipeline.

Plain-language overview: a typed question goes through a medical/advice
check (guard.py), then a keyword parser (rules.py) and, only if that isn't
sure, the local model (llm_parser.py). Both parsers only fill in a small
form -- which metric, which activity, which time range. pipeline.py checks
the form and answers it with data/queries.py's run(), the same code the
Dashboard uses, and answers.py writes every sentence the user reads. The
model never writes a number or a word that's shown.
"""
