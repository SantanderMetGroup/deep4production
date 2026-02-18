from setuptools import setup, find_packages

setup(
    name='d4p',
    version='0.0',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'd4p-train = deep4production.console.main_train:main',
            'd4p-downscale = deep4production.console.main_downscale:main',
            'd4p-datasets-inspect = deep4production.console.main_inspect:main',
            'd4p-datasets-create = deep4production.console.main_create:main'
        ]
    }
)

