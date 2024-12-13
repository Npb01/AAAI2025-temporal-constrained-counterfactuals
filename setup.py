from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import urllib.request
import tarfile

# Constants
MONA_URL = "http://www.brics.dk/mona/download/mona-1.4-18.tar.gz"
MONA_DIR = "mona-1.4"
MONA_TAR = "mona-1.4-18.tar.gz"

class BuildExt(build_ext):
    def run(self):
        self.download_and_install_mona()
        super().run()

    def download_and_install_mona(self):
        # Download the MONA package
        if not os.path.exists(MONA_TAR):
            print(f"Downloading MONA from {MONA_URL}...")
            urllib.request.urlretrieve(MONA_URL, MONA_TAR)

        # Extract the tar.gz file
        if not os.path.exists(MONA_DIR):
            print(f"Extracting {MONA_TAR}...")
            with tarfile.open(MONA_TAR, "r:gz") as tar:
                tar.extractall()

        # Compile and install MONA
        os.chdir(MONA_DIR)
        print(f"Configuring MONA in {os.getcwd()}...")
        subprocess.check_call(["./configure"])

        print("Compiling MONA...")
        subprocess.check_call(["make"])

        print("Installing MONA...")
        subprocess.check_call(["make", "install"])

        os.chdir("..")


# Setup configuration
setup(
    name='src',
    version='0.1',
    packages=find_packages(),
    cmdclass={'build_ext': BuildExt},
)
