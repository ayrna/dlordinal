import os
import re
import sys

from check_version_consistency import get_versions_set

release_title = os.environ.get("RELEASE_TITLE")
if not release_title:
    print(
        "❌ RELEASE_TITLE environment variable not set. Please check that you have set "
        "the release title appropriately."
    )
    sys.exit(1)

versions = get_versions_set()
matches = re.findall(r"([0-9]+\.[0-9]+\.[0-9]+)", release_title)
release_version = matches[0] if matches else None
versions.add(release_version)

print(f"Release title: {release_title}")
print(f"Versions found in files: {versions}")
print(f"Release version: {release_version}")

if len(versions) != 1:
    print("❌ Version mismatch detected!")
    sys.exit(1)

print("✅ Versions match.")
