import glob
from setuptools import setup, find_packages

setup(
    name="pybenzinaconcat",
    version="0.4.3",
    packages=find_packages(exclude=["test_*"]),
    url="https://github.com/satyaog/pybenzinaconcat",
    license="The MIT License",
    author="Satya Ortiz-Gagne",
    author_email="satya.ortiz-gagne@mila.quebec",
    description="",
    install_requires=["jug",
                      "pillow",
                      "numpy",
                      "pybenzinaparse @ git+https://github.com/satyaog/pybenzinaparse.git@0.2.2#egg=pybenzinaparse-0.2.2"],
    extras_require={"h5py": ["h5py"]},
    tests_require=["jug >= 2.1.1", "h5py", "pytest"],
    long_description="",
    data_files=[("tests", glob.glob("test_datasets/*", recursive=True))]
)
