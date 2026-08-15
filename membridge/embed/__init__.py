"""Query-side embedding, in the space the migrated memories already occupy.

Not a general embedding abstraction. The store's space is fixed by what was
migrated into it, so there is exactly one correct answer here and this module's
job is to produce it without dragging torch into a Lambda package.
"""

from membridge.embed.minilm import (
    MAX_TOKENS,
    MODEL_DIM,
    MODEL_DIR_ENV,
    MODEL_NAME,
    EncoderUnavailable,
    MiniLMEncoder,
    default_encoder,
)

__all__ = [
    "MAX_TOKENS",
    "MODEL_DIM",
    "MODEL_DIR_ENV",
    "MODEL_NAME",
    "EncoderUnavailable",
    "MiniLMEncoder",
    "default_encoder",
]
