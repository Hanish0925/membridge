"""HTTP surfaces for MemBridge.

Separate from `membridge.cli` because the constraints are different: a CLI
process handles one request and exits, so it may import freely and leak
whatever it likes. A Lambda container handles thousands and must hold its
expensive objects -- the ONNX session, the database connection -- across
invocations while surviving both going stale.
"""
