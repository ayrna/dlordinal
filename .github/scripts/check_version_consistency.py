import re
import sys
from pathlib import Path


def extract_version_from_pyproject():
    content = Path("pyproject.toml").read_text()
    match = re.search(r'version\s*=\s*"(.+?)"', content)
    return match.group(1) if match else None


def extract_version_from_init():
    content = Path("dlordinal/__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*["\'](.+?)["\']', content)
    return match.group(1) if match else None


def extract_versions_from_readme():
    content = Path("README.md").read_text()
    matches = re.findall(r"v([0-9]+\.[0-9]+\.[0-9]+)", content, re.IGNORECASE)
    return matches


def get_versions_set():
    pyproject_version = extract_version_from_pyproject()
    init_version = extract_version_from_init()
    readme_version = extract_versions_from_readme()

    print(f"pyproject.toml: {pyproject_version}")
    print(f"__init__.py: {init_version}")
    print(f"README.md: {readme_version}")

    versions = {pyproject_version, init_version, *readme_version}

    return versions


if __name__ == "__main__":
    versions = get_versions_set()

    if None in versions or len(versions) != 1:
        print("❌ Version mismatch detected!")
        sys.exit(1)

    print("✅ Versions match.")
