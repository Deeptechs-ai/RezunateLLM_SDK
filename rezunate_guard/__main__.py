"""Entry point for `python -m rezunate_guard`.

The hook is registered by absolute command rather than by name, so it keeps working
wherever the package is installed. See `cli.HOOK_COMMAND`.
"""

import sys

from rezunate_guard.cli import main

if __name__ == "__main__":
    sys.exit(main())
