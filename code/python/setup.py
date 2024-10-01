from setuptools import setup, find_packages

setup(
    name='balanced_neural_odes',
    version='0.3',
    packages=find_packages(
        include=[
            'networks*',
            'data_generation*',
            'utils*',
            # 'config.py',
            # 'filepaths.py'
        ],
        exclude=[
            'torchdiffeq'
        ]
    ),
)