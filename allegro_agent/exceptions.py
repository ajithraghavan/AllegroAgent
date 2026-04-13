class FrameworkError(Exception):
    """Base exception for the AI Agent Framework."""
    pass


class ProviderError(FrameworkError):
    """Raised when a provider encounters an error."""
    pass


class ProviderNotFoundError(FrameworkError):
    """Raised when a requested provider is not registered."""
    pass


class InvalidModelFormatError(FrameworkError):
    """Raised when the model string doesn't follow 'provider:model' format."""
    pass


class ToolError(FrameworkError):
    """Raised when a tool encounters an error during execution."""
    pass
