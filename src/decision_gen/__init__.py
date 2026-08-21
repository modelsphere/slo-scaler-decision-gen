"""decision-gen — LLM inference scale-decision operator.

Periodically computes, per LLM service, how many replicas should be
active based on SLO signals and cluster capacity, and serves the result
over HTTP for an external executor to apply.
"""

__version__ = "0.1.0"
