
# Massive Multi-Agent Worm Simulator (MMA-WORMSIM)

## Prerequisites

Ensure you have the following software installed before proceeding:
- **CUDA Toolkit**: 12.3
- **GCC**: 11.4.0
## Setting Up Environment Variables

### Step 1: Verify Installed Versions

You can verify the installed versions of `gcc` and `nvcc` using the following commands:

```bash
gcc --version
nvcc --version
```

Make sure the versions match the ones required for the project.

### Step 2: Setting the Environment Variables

Add the following lines to your shell configuration file (e.g., `.bashrc`, `.zshrc`) to set the required environment variables:

```bash
# Set environment variables for GCC and NVCC
export PATH=/usr/local/cuda-X.X/bin:$PATH  # Update with the correct CUDA version
export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-X.X                 # Update with your GCC path and version
export CXX=/usr/bin/g++-X.X                # Update with your G++ path and version
```

After modifying the file, apply the changes by running:

```bash
source ~/.bashrc  # or ~/.zshrc depending on your shell
```

### Step 3: Verify the Setup

To confirm that the environment variables are set correctly, run:

```bash
echo $PATH
echo $LD_LIBRARY_PATH
echo $CC
echo $CXX
```

### Step 4: Compiling the CUDA Simulator

Once the environment is configured, compile the CUDA simulator by navigating to the project directory and running the following command:

```bash
nvcc main.cu -o main -lm --expt-relaxed-constexpr
```
