import inspect
from typing import Any


def get_init_signature(callable) -> dict[str, Any]:
    """Get the signature of the function"""
    signature = inspect.signature(callable)

    init_signature = {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in signature.parameters.items()
        if k not in {"self", "args", "kwargs"}
    }

    return init_signature
