__version__ = "0.1.0"

# Ensure torch.library ops are registered on import.
from forge_cute_py import ops as _ops  # noqa: F401
