import launch
import os

print()
print("Running install.py for sd-prompt-pinning..")

if not launch.is_installed('deap'):
    launch.run_pip("install deap==1.4.1", "requirements for sd-prompt-pinning: deap")

if not launch.is_installed('matplotlib'):
    launch.run_pip("install matplotlib==3.8.0", "requirements for sd-prompt-pinning: matplotlib")

path = os.path.dirname(os.path.realpath(__file__))

# current as of 12/18/2023
launch.git_clone(
    "https://github.com/NVlabs/flip.git",
    os.path.join(path, "flip"),
    "flip",
    "cd2166b28d0549e9a4a5fea0fdcb9ebf7a48a4cd")

print("Ran install.py for sd-prompt-pinning")
print()

