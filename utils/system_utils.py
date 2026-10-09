from errno import EEXIST
from os import makedirs, path


# ---- filesystem utilities ----
def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise
