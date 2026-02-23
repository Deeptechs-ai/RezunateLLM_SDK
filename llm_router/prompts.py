"""Standalone template rendering utility."""

import re

_TEMPLATE_VAR_RE = re.compile(r"\{\{([a-zA-Z_]\w*)\}\}")


def render_prompt(content: str, variables: dict[str, str] | None = None) -> str:
    """Replace {{variable}} placeholders in a prompt template.

    Args:
        content: The prompt template string.
        variables: Mapping of variable names to values.

    Returns:
        The rendered prompt string.

    Raises:
        ValueError: If the template contains variables not provided in `variables`.
    """
    if not variables:
        variables = {}

    required = set(_TEMPLATE_VAR_RE.findall(content))
    missing = required - set(variables.keys())
    if missing:
        raise ValueError(f"Missing template variables: {', '.join(sorted(missing))}")

    return _TEMPLATE_VAR_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), content)
