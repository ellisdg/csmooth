import logging
import sys

logger = logging.getLogger("csmooth")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)  # Use sys.stderr for STDERR
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
