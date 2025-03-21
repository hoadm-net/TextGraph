from os import path, getcwd, pardir


def get_data_path(data) -> str:
    current_dir = getcwd()
    base_dir = path.abspath(path.join(current_dir, pardir))
    return path.join(base_dir, 'datasets', data)
