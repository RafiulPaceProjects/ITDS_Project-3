import warnings

# Streamlit imports urllib3 before executing app.py; on some macOS Python builds
# urllib3 v2 emits a noisy LibreSSL warning. It's non-fatal for this project.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*LibreSSL.*",
)
