"""Post-hoc analysis of a scored factorial: the per-prompt count table and the four
reported metrics computed from it.

Every statistic here is a function of one count tensor and an index array over prompts.
The point estimate, a bootstrap replicate, a jackknife term and a Manski corner are the
same function called on different rows, or on a copy whose malformed rows were resolved
the other way -- so each metric has exactly one implementation.
"""
