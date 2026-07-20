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


def require_dataset_runtime():
    """Fail early when dataset download dependencies are not import-compatible."""
    try:
        import_module("urllib3._request_methods")
        import_module("requests")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The installed dataset download stack is too old or incomplete. "
            "Refresh the experiment environment with: "
            "python -m pip install --upgrade --force-reinstall urllib3 requests "
            "or refresh all project dependencies with: "
            "python -m pip install --upgrade -r requirements.txt"
        ) from exc
