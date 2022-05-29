#  Copyright (c) modalic 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import logging


class CustomFormatter(logging.Formatter):
    r"""Custom formatting object."""

    # grey = "\x1b[38;20m"
    green = "\x1b[32m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    @staticmethod
    def custom_format(color, reset):
        r"""Returns a custom format string."""
        return (
            "%(name)s: %(asctime)s "
            + color
            + " %(levelname)s"
            + reset
            + " : %(message)s"
            + reset
        )

    FORMATS = {
        logging.DEBUG: custom_format.__func__(yellow, reset),
        logging.INFO: custom_format.__func__(green, reset),
        logging.WARNING: custom_format.__func__(yellow, reset),
        logging.ERROR: custom_format.__func__(red, reset),
        logging.CRITICAL: custom_format.__func__(bold_red, reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("modalic")

# logger configuration
logger.setLevel(level=logging.INFO)
# handler configuration
handler = logging.StreamHandler()
handler.setLevel(level=logging.INFO)
handler.setFormatter(CustomFormatter())

logger.addHandler(handler)
