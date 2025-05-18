from setuptools import setup, find_packages

setup(
    name='music_source_separation_training',
    version='0.1.0',
    description='Music source separation training tools',
    author='Unknown',
    packages=find_packages(include=[
        'models*',
        'scripts*',
        'configs*',
        'tests*'
    ]),
    py_modules=[
        'dataset',
        'ensemble',
        'inference',
        'metrics',
        'train',
        'train_accelerate',
        'utils',
        'valid'
    ],
    install_requires=[

    ],
    include_package_data=True,
    zip_safe=False,
)
