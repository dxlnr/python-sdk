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


class Monitor(object):
    r"""Monitoring the Modalic client while training.

    Args:
        log_name: logging name that gets printed by default.
    """

    def __init__(self, log_name: str = "modalic"):
        self.logger = logging.getLogger(log_name)

        # logger configuration
        self.logger.setLevel(level=logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(level=logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(name)s: %(asctime)s %(levelname)s | %(message)s")
        )
        self.logger.addHandler(handler)

    # def configure(self):
    #     pass
