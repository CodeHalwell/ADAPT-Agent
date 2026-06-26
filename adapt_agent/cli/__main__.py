"""Enable ``python -m adapt_agent.cli`` to invoke the CLI."""

import sys

from adapt_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
