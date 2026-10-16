"""Custom domain-specific exceptions for NEXUS-9."""

class Nexus9Error(Exception):
    """Base exception for all NEXUS-9 errors."""
    pass

class DataValidationError(Nexus9Error):
    """Raised when data fails schema validation or contains illicit leakage."""
    pass

class DataLeakageError(DataValidationError):
    """Raised when early-warning features include outcome definitions."""
    pass

class ModelNotFoundError(Nexus9Error):
    """Raised when an un-registered model family is requested."""
    pass

class SimulationError(Nexus9Error):
    """Raised when simulation conditions or transitions violate invariant checks."""
    pass

class CausalInferenceError(Nexus9Error):
    """Raised when causal assumptions or graph structure are invalid."""
    pass

class AgentExecutionError(Nexus9Error):
    """Raised when an autonomous agent encounters an unrecoverable failure."""
    pass
