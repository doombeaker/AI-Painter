import subprocess
import importlib.util
import os
import sys

from ui import create_ui

dir_repos = "repositories"
dir_extensions = "extensions"
python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")

def repo_dir(name):
    return os.path.join(dir_repos, name)

def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C {dir} rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C {dir} fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C {dir} checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C {dir} checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

def run(command, desc=None, errdesc=None, custom_env=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None

def prepare_enviroment():
    requirements_command = f"{python} -m pip install -r requirements.txt"
    run(requirements_command, "Installing requirments.txt", "Couldn't installing requirments.txt")

    of_diffusers_repo = "https://github.com/Oneflow-Inc/diffusers.git"
    git_clone(of_diffusers_repo, repo_dir('diffusers'), "OneFlow Diffusers", "495155448aaf7391a4edb3ffcefced015b4080f2")
    diffusers_command = f"cd {repo_dir('diffusers')} && python -m pip install -e .[oneflow]"
    if not is_installed("diffusers"):
        run(diffusers_command, "Installing oneflow-diffusers", "Couldn't install oneflow-diffusers")

    of_transformers_repo = "https://github.com/Oneflow-Inc/transformers.git"
    git_clone(of_transformers_repo, repo_dir("transformers"), "OneFlow Transformers", "f7a52b04a0035aaf4b48c999a170ef630d3af9b6")
    transformers_command = f"cd {repo_dir('transformers')} && python -m pip install -e ."
    if not is_installed("transformers"):
        run(transformers_command, "Installing oneflow-transformers", "Couldn't install oneflow-transformers")

if __name__ == "__main__":
    prepare_enviroment()
    create_ui()