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

"""Aggregation server related API."""

import subprocess

from modalic.server.api import find_bin_path


def run_server(cfg_path: str = "") -> None:
    r"""Runs the Federated Learning aggregation server.

    :param cfg_path: Path to an external .toml configuration file.

    :Example:
    >>> import argparse
    >>> parser = argparse.ArgumentParser(description="Server arguments.")
    >>> parser.add_argument("--cfg", type=str, help="configuration file (path)")
    >>> args = parser.parse_args()

    >>> modalic.run_server(args.cfg)
    """
    command = [find_bin_path()]
    if cfg_path and cfg_path.strip():
        command.extend(["-c", cfg_path])
    try:
        subprocess.run(command, shell=False)
    except subprocess.CalledProcessError:
        return
