import os
import sys

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infer import main

if __name__ == "__main__":
    main()
