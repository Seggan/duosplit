from sirilpy import SirilInterface
from sirilpy.utility import download_with_progress

siril = SirilInterface()
siril.connect()

def get_runtime_version():
    import platform
    if platform.system() == "Linux":
        if platform.machine() == "x86_64":
            return "duosplit-x86_64-unknown-linux-gnu"
        elif platform.machine() == "aarch64":
            return "duosplit-aarch64-unknown-linux-gnu"
    elif platform.system() == "Windows" and '64' in platform.machine():
        return "duosplit-x86_64-unknown-linux-gnu"
    siril.error_messagebox(
        "This script only provides a runtime for Linux x86_64, Linux ARM64, and Windows x86_64 systems. "
        "If you do not have such a system, you will need to download the runtime from "
        "'https://github.com/Seggan/duosplit', compile it, and place the compiled executable in "
        f"'{siril.get_siril_userdatadir()}' as 'duosplit'."
    )
    exit(1)

def get_latest_release():
    import requests
    api_url = "https://api.github.com/repos/Seggan/duosplit/releases/latest"
    response = requests.get(api_url)
    response.raise_for_status()
    release_info = response.json()
    return release_info

def get_runtime_asset(error):
    release_info = get_latest_release()
    assets = release_info.get("assets", [])
    runtime_version = get_runtime_version()
    for asset in assets:
        if asset["name"] == runtime_version:
            return asset
    else:
        if error:
            siril.error_messagebox(
                "Could not find a suitable duosplit runtime for your system in the latest release."
            )
            exit(1)
        return None


from pathlib import Path

RUNTIME_PATH = Path(siril.get_siril_userdatadir()) / "duosplit"
if not RUNTIME_PATH.exists():
    siril.log("Duosplit runtime not found. Downloading...")
    asset = get_runtime_asset(error=True)
    download_url = asset["browser_download_url"]
    download_with_progress(
        siril,
        download_url,
        str(RUNTIME_PATH)
    )

from hashlib import sha256
current_hash = "sha256:" + sha256(RUNTIME_PATH.read_bytes()).hexdigest()
asset = get_runtime_asset(error=False)
if asset:
    expected_hash = asset["digest"]
    if current_hash != expected_hash:
        siril.log("A new version of duosplit is available. Downloading update...")
        download_url = asset["browser_download_url"]
        download_with_progress(
            siril,
            download_url,
            str(RUNTIME_PATH)
        )
    else:
        siril.log("Duosplit runtime is up to date.")