"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import os # for os related ops


def create_folders_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        _path = path.split("/")[:-1]
        if path.startswith("/"):
            _path[0] = "/"

        os.makedirs(os.path.join(*_path), exist_ok=True)
