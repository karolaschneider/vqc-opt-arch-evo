#!/bin/bash

# SLURM configuration
# Use #SBATCH <option>=<value> to configure. Do not uncomment lines with #SBATCH
# Ref: https://doku.lrz.de/slurm-workload-manager-10745909.html
# Use sinfo to see all availiable partitions
# For LMU Cip SLURM, use --partition=NvidiaAll if you need nodes with a gpu

#SBATCH --mail-user=user@email.com  # Replace with your email
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=All
#SBATCH --export=NONE

# Environment Variables
# export WANDB_MODE="disabled" # Use if you want to disable wandb logging
export WANDB_SILENT="true"

# Initialize pyenv in the SLURM environment
export PYENV_ROOT="$HOME/.pyenv" # pyenv needs to be installed in the home directory
export PATH="$PYENV_ROOT/bin:$PATH"

# Check if pyenv is installed
if command -v pyenv 1>/dev/null 2>&1; then
    # Setup pyenv shell
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
else
    echo "pyenv is not available in the SLURM environment. Exiting."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null)
install_python=false
if [[ -z "$python_version" || "$python_version" < "3.11" ]]; then
    echo "Python 3.11 or higher is not installed. Setting flag to install..."
    install_python=true
fi

# Install Python 3.11 if necessary
if [ "$install_python" = true ]; then
    pyenv install 3.11.0
    pyenv global 3.11.0
fi

# Check if Python version was successfully installed
if [ "$install_python" = true ]; then
    python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null)
    if [[ -z "$python_version" || "$python_version" != "3.11" ]]; then
        echo "Failed to install Python 3.11. Exiting."
        exit 1
    fi
fi

# Create a fresh virtual environment
pyenv virtualenv env 1>/dev/null 2>&1
pyenv activate env 1>/dev/null 2>&1

# Check the exit status of the pyenv activate command
if [ $? -ne 0 ]; then
    echo "\033[31mFailed to activate the virtual environment using pyenv. Exiting.\033[0m"
    exit 1
fi

# Install packages listed in requirements.txt
pip install -qr requirements.txt

# Runs the script
python src/main.py $@