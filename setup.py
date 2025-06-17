import io
import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install


def check_java_version():
    try:
        result = subprocess.run(['java', '-version'], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        version_str = result.stderr.decode() if result.stderr else result.stdout.decode()
        
        # Extract version number
        version = version_str.split('"')[1] if '"' in version_str else ''
        if version:
            major_version = int(version.split('.')[0])
            if major_version < 21:
                print("WARNING: Java version must be 21 or higher. Current version:", version)
                print("Please install Java 21 from:")
                print("  - Ubuntu/Debian: sudo apt install openjdk-21-jdk")
                print("  - macOS: brew install openjdk@21")
                print("  - Windows: https://www.oracle.com/java/technologies/downloads/#java21")
                sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Java is not installed. Please install Java 21.")
        print("Installation instructions:")
        print("  - Ubuntu/Debian: sudo apt install openjdk-21-jdk")
        print("  - macOS: brew install openjdk@21")
        print("  - Windows: https://www.oracle.com/java/technologies/downloads/#java21")
        sys.exit(1)


class CustomInstall(install):
    def run(self):
        check_java_version()
        install.run(self)


def read(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="autometrics",
    version=read("VERSION"),
    description="Package for the autometrics project.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    cmdclass={'install': CustomInstall},
)