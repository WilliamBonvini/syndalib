from setuptools import find_packages, setup

setup(
    name="syndalib",
    packages=find_packages(
        include=[
            "syndalib",
            "syndalib.utils",
            "syndalib.makers",
        ]
    ),
    version="0.1.15",
    description="synthetic point cloud library",
    author="William Bonvini",
    license="MIT",
    install_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
)
