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
import os
import subprocess

from tools.env import check_submodules


def prebuild_w_maturin(
    root_path: str, m_path: str = "modules/mosaic/mosaic-bindings/python"
):
    r"""."""
    path = os.path.join(root_path, m_path)
    # Change the current working directory.
    try:
        os.chdir(path)
    except NotADirectoryError:
        print(f"{path} is not a directory")
    except PermissionError:
        print(f"You do not have permissions to change to {path}")

    # Build with maturin.
    try:
        subprocess.check_call(["maturin", "develop", "--release"], cwd=path)
    except Exception as err:
        print(f"prebuilding python binding failed: {err}.")

    # go back.
    os.chdir(root_path)

    # static_lib_file = "/target/release/libmosaic_python_sdk.so"
    # # copy to destination.
    # try:
    #     subprocess.check_call(
    #         [
    #             "scp",
    #             path + static_lib_file,
    #             root_path + "/modalic/client/mosaic_python_sdk.so",
    #         ],
    #         cwd=path,
    #     )
    # except Exception as err:
    #     print(f"copying {path + static_lib_file} failed: {err}.")


def build_deps(root_path: str):
    r"""."""
    check_submodules(root_path)

    prebuild_w_maturin(root_path)
