import pathlib
import os

def get_paper_path(*args):
    # _path = "../../../../../LaTeX/ICLR 2025 Template/figures"
    # choose root path depending on how far from balanced_neural_odes the script is
    _cwd = os.getcwd()
    # count how many times we need to go up
    _up = 0
    print('finding path that contains "code", then going up one level to find "LaTeX"')
    while not pathlib.Path(_cwd).joinpath("code").exists():
        _cwd = os.path.dirname(_cwd)
        print(_cwd)
    # _cwd = os.path.dirname(_cwd)
    _path = pathlib.Path(_cwd).joinpath("LaTeX/ICLR 2025 Template/figures")
    for i, value in enumerate(args):
        _path = _path / value
        if i < len(args) - 1:
            _path.resolve().mkdir(exist_ok=True)
    return _path