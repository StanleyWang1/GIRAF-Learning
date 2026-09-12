"""Machine-readable status of provisional learning integration points."""

from typing import Final

# Flip only after the observation/action representation is intentionally frozen
# across collection, training, and deployment.
POLICY_CONTRACT_FINAL: Final = False
