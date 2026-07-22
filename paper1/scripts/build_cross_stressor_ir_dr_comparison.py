#!/usr/bin/env python3
"""Deprecated wrapper for the released IR/DR-v1 builder entry point.

New derived artifacts use the IR/SR-v2 schema and are built by
``build_cross_stressor_ir_sr_comparison.py``.
"""

if __package__:
    from .build_cross_stressor_ir_sr_comparison import main
else:  # Preserve the historical direct-script invocation.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper1.scripts.build_cross_stressor_ir_sr_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
