#!/usr/bin/env python3
"""Deprecated wrapper for the former IR/DR Figure 1 script name.

The current metric name is Separation Rate (SR); use
``plot_acpc_ir_sr_overview.py`` for new workflows.
"""

if __package__:
    from .plot_acpc_ir_sr_overview import main
else:  # Preserve the historical direct-script invocation.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper1.scripts.plot_acpc_ir_sr_overview import main


if __name__ == "__main__":
    raise SystemExit(main())
