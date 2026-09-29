"""``python -m neural_flow``: same as the ``neural-flow`` command."""
import sys

from .cli import main

sys.exit(main())
