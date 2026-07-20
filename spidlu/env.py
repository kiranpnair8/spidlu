"""Runtime dependency checks for experiment entrypoints."""

from importlib import import_module


def require_huggingface_runtime():
    """Fail early when the Hugging Face stack is not import-compatible."""
    try:
        import_module("huggingface_hub.errors")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The installed Hugging Face Hub package is too old or incomplete. "
            "Refresh the experiment environment with: "
            "python -m pip install --upgrade -r requirements.txt"
        ) from exc
