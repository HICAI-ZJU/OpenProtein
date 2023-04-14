from .ec import ProteinFunctionDecoder
from .flip import SequenceRegressionDecoder
from .tape import SequenceClassificaitonDecoder, SequenceToSequenceClassificaitonDecoder, ProteinContactMapDecoder

__all__ = [
    "ProteinFunctionDecoder",
    "SequenceRegressionDecoder",
    "SequenceClassificaitonDecoder",
    "SequenceToSequenceClassificaitonDecoder",
    "ProteinContactMapDecoder"
]