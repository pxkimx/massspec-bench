"""MassSpec Bench server package.

Point every TLS client at certifi's CA bundle before anything can open a connection.

The Mac app runs on a python.org framework build, which ships no CA certificates of its own and does
not read the system keychain: `urllib` there fails with CERTIFICATE_VERIFY_FAILED for every https
download. `requests` hides this because it uses certifi already — which is why GSEApy's gene sets
downloaded fine while decoupler's PROGENy/CollecTRI networks (urllib, via omnipath) never could, and
reported themselves as "needs internet" on a machine that had internet.

setdefault, so a user or sysadmin who has pointed these at their own bundle keeps it.
"""
import os

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:  # noqa: BLE001 - never let this stop the server from starting
    pass
