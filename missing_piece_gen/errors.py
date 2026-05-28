"""Pipeline error hierarchy for missing-piece-gen."""


class PipelineError(Exception):
    """Base error for all pipeline failures."""


class DetectionError(PipelineError):
    """Image segmentation could not detect pieces or the missing slot."""


class EdgeExtractionError(PipelineError):
    """Edge profile extraction failed for a piece."""


class InferenceError(PipelineError):
    """Could not compute a valid missing piece shape."""


class ModelGenerationError(PipelineError):
    """3D model generation or export failed."""
