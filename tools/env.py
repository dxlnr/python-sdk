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
import sys
import os
import subprocess


def get_submodule_folders(root_path: str, modules_path: str = "modules"):
    r"""Returns git submodules folder path.

    :param root_path: Root of the library.
    :param modules_path: Path where all submodules are present.
    """
    git_modules_path = os.path.join(root_path, ".gitmodules")
    default_modules_path = [
        os.path.join(modules_path, name)
        for name in [
            "mosaic",
        ]
    ]
    if not os.path.exists(git_modules_path):
        return default_modules_path
    with open(git_modules_path) as f:
        return [
            os.path.join(root_path, line.split("=", 1)[1].strip())
            for line in f.readlines()
            if line.strip().startswith("path")
        ]


def check_submodules(root_path: str):
    r"""initializes the submodules and keeps them in sync."""

    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            print("Could not find any of {} in {}".format(", ".join(files), folder))
            print("Did you run 'git submodule update --init --recursive ?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (
            os.path.isdir(folder) and len(os.listdir(folder)) == 0
        )

    folders = get_submodule_folders(root_path)

    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(" --- Trying to initialize submodules")
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"], cwd=root_path
            )
            print(" --- Submodule initialized.")
        except Exception:
            print(" --- Submodule initalization failed.")
            print("Please run:\n\tgit submodule update --init --recursive")
            sys.exit(1)
    for folder in folders:
        check_for_files(folder, ["mosaic/Cargo.toml"])
