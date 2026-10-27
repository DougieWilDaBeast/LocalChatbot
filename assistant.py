#!/usr/bin/env python3
"""
NUC Offline Voice Assistant. The code lives in the nuc_assistant package;
this script is the entry point the README and the systemd service use.

Run:
  python3 assistant.py
  python3 assistant.py --debug
  python3 assistant.py --no-wake-word   (skip wake word, press Enter to speak)
"""

from nuc_assistant.main import main

if __name__ == "__main__":
    main()
