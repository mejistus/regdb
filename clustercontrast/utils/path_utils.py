from __future__ import print_function, absolute_import

import os.path as osp


RED = "\033[31m"
RESET = "\033[0m"


def warn_if_relative_path(path, label="--data-dir"):
    expanded = osp.expanduser(path)
    if not osp.isabs(expanded):
        print(
            "{}WARNING: 您使用的是相对路径 {}=\"{}\"；请从仓库根目录运行，或确认数据已放在该相对路径下。{}".format(
                RED, label, path, RESET
            )
        )


def warn_relative_data_dir(data_dir):
    warn_if_relative_path(data_dir, "--data-dir")
