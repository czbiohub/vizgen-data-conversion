from setuptools import setup, find_packages

setup(
    name='yourscript',
    version='0.1.0',
    py_modules=['cli'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'vizgen-tools = cli:cli',
        ],
    },
)
