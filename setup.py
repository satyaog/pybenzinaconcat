import glob
from setuptools import setup, find_packages

setup(
    name="pybenzinaconcat",
    version="0.1.0",
    packages=find_packages(exclude=["test_*"]),
    url="https://github.com/satyaog/pybenzinaconcat",
    license="The MIT License",
    author="Satya Ortiz-Gagne",
    author_email="satya.ortiz-gagne@mila.quebec",
    description="",
    install_requires=["pybenzinaparse @ git+https://github.com/satyaog/pybenzinaparse.git@0.2.1#egg=pybenzinaparse-0.2.1",
                      "pillow>=6.2.0"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest>=5.0.1"],
    long_description="",
    data_files=[("tests", glob.glob("test_datasets/*", recursive=True))]
)
