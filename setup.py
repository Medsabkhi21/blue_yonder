import subprocess
import sys

def install_requirements():
    """
    The install_requirements function installs the requirements.txt file using pip.
    
    :return: Nothing
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()