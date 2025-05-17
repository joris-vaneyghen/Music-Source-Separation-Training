from setuptools import setup, find_packages

setup(
    name='mms_training',
    version='0.1.0',
    description='Audio source separation tool for isolating vocals and instruments',
    author='Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/',
    author_email='',
    packages=find_packages(),
    install_requires=[
        'librosa',
        'torch',
        'soundfile',
        'numpy',
        'tqdm',
        'matplotlib',
        'argparse'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'mms_training=inference:proc_folder',
        ],
    },
)
