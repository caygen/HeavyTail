from setuptools import setup
wtih open("README.MD", "r") as fh:
    long_description = fh.read()
setup(
    name = 'HeavyTail',
    version = '0.0.1',
    description = 'FirstVersion: SE or AD',
    py_modules = ["helloworld"],
    package_dir = {'': 'src'},

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.0",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Academic Free License (AFL)",
        "Operating System :: OS Independent",
    ],
    long_description = long_description,
    long_description_content_type = "text/markdown",

    
)