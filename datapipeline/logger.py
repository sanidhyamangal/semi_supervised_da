import logging
import logging.handlers
import sys
import os
import inspect
from datetime import date

consoleLoggingLevel = logging.DEBUG


loggerName = 'datapipeline'

logger = logging.getLogger(loggerName)
logger.setLevel(logging.DEBUG)


ch = logging.StreamHandler(sys.stdout)

# Set console logging level
ch.setLevel(consoleLoggingLevel)

formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
ch.setFormatter(formatter)

# logger.addHandler(fh)
logger.addHandler(ch)

logger.debug('Started logging')
