"""agent-fast-decider: System-1/System-2 support-ticket triage."""

from .decider import FastLocalDecider, RealDecider, load_decider
from .triage import TriageAgent, load_tickets

__all__ = ["FastLocalDecider", "RealDecider", "TriageAgent", "load_decider", "load_tickets"]
__version__ = "0.1.0"
