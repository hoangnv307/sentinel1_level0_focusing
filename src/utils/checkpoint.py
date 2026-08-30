"""Thin policy adapter for :class:`joblib.Memory`."""

from typing import Any, Callable, Literal


CachePolicy = Literal["reuse", "refresh", "readonly", "off"]
_CACHE_POLICIES = {"reuse", "refresh", "readonly", "off"}


def cached_call(
    cached_function: Any,
    original_function: Callable[..., Any],
    policy: CachePolicy,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if policy not in _CACHE_POLICIES:
        raise ValueError(
            f"Unknown cache policy {policy!r}; expected one of {sorted(_CACHE_POLICIES)}"
        )
    if policy == "off":
        return original_function(*args, **kwargs)
    if policy == "readonly":
        if not cached_function.check_call_in_cache(*args, **kwargs):
            raise FileNotFoundError("Required cache checkpoint does not exist")
        return cached_function(*args, **kwargs)
    if policy == "refresh":
        result, _ = cached_function.call(*args, **kwargs)
        return result
    return cached_function(*args, **kwargs)
