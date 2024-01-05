import launch
import os
import subprocess

print()
print("Running install.py for sd-prompt-pinning..")

path = os.path.dirname(os.path.realpath(__file__))

# # trailing '#' because a '--' option appears to be automatically passed
# launch.run_pip("uninstall deap #", "uninstall deap..")

if not launch.is_installed('deap'):
    deap_path = os.path.join(path, "deap")
    launch.git_clone(
        "https://github.com/michaeljklein/deap",
        deap_path,
        "deap",
        "6f7c55c833d0c007f4223efdcc17f8d90ca9922b")

    # launch.run_pip("install deap==1.4.1", "requirements for sd-prompt-pinning: deap")
    # launch.run_pip(f"install --force-reinstall -e '{deap_path}'", "requirements for sd-prompt-pinning: deap")
    launch.run_pip(f"install -e '{deap_path}'", "requirements for sd-prompt-pinning: deap")

# else:
#     raise ValueError("failed to install deap")


if not launch.is_installed('SciPy'):
    launch.run_pip("install SciPy==1.11.4", "requirements for sd-prompt-pinning: UMAP: SciPy")

if not launch.is_installed('scikit-learn'):
    launch.run_pip("install scikit-learn==1.3.2", "requirements for sd-prompt-pinning: UMAP: scikit-learn")

if not launch.is_installed('numba'):
    launch.run_pip("install numba==0.58.1", "requirements for sd-prompt-pinning: UMAP: numba")

if not launch.is_installed('pynndescent'):
    launch.run_pip("install pynndescent==0.5.11", "requirements for sd-prompt-pinning: UMAP: pynndescent")

if not launch.is_installed('umap-learn'):
    launch.run_pip("install umap-learn==0.5.5", "requirements for sd-prompt-pinning: UMAP")

if not launch.is_installed('fast-hdbscan'):
    launch.run_pip("install fast-hdbscan==0.1.3", "requirements for sd-prompt-pinning: fast-hdbscan")

if not launch.is_installed('matplotlib'):
    launch.run_pip("install matplotlib==3.8.0", "requirements for sd-prompt-pinning: matplotlib")

# current as of 12/18/2023
launch.git_clone(
    "https://github.com/NVlabs/flip.git",
    os.path.join(path, "flip"),
    "flip",
    "cd2166b28d0549e9a4a5fea0fdcb9ebf7a48a4cd")

print("Ran install.py for sd-prompt-pinning")
print()

