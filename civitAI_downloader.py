import os
import re
import sys
import getpass
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Thread-local storage — each worker gets its own Session (thread-safe)
_thread_local = threading.local()

VALID_EXTENSIONS = {".pt", ".safetensors"}
REQUEST_TIMEOUT = (10, 60)   # (connect, read) seconds
MIN_CHECKPOINT_SIZE = 2 * 1024 ** 3  # 2 GB — safetensors >= this → models/


def get_session(token: str) -> requests.Session:
    """Return a thread-local Session with auth header pre-set."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        if token:
            _thread_local.session.headers["Authorization"] = f"Bearer {token}"
    return _thread_local.session


def sanitize_filename(name: str) -> str:
    """Strip directory components and dangerous characters from a filename.

    Raises ValueError if the result is unusable.
    """
    # Strip any directory traversal
    name = os.path.basename(name)
    if ".." in name:
        raise ValueError(f"Path traversal in filename: {name!r}")
    # Remove control characters and shell-dangerous chars
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', "_", name)
    name = name.strip(". ")
    if not name:
        raise ValueError("Filename is empty after sanitization")
    return name


def destination_folder(filename: str, file_size: int) -> str:
    """Determine target subfolder from extension and file size.

    .pt                        → embeddings/
    .safetensors  < 2 GB       → loras/
    .safetensors  >= 2 GB      → models/   (likely a checkpoint)
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pt":
        return "embeddings"
    if file_size >= MIN_CHECKPOINT_SIZE:
        return "models"
    return "loras"


def file_already_exists(filename: str) -> bool:
    for folder in ("embeddings", "loras", "models"):
        if os.path.exists(os.path.join(folder, filename)):
            return True
    return False


def download_one(filename: str, url: str, token: str) -> bool:
    """Download a single file. Returns True on success, False on failure."""
    try:
        filename = sanitize_filename(filename)
    except ValueError as e:
        print(f"  SKIP  {e}")
        return False

    ext = os.path.splitext(filename)[1].lower()
    if ext not in VALID_EXTENSIONS:
        print(f"  SKIP  {filename} — unsupported extension ({ext})")
        return False

    if file_already_exists(filename):
        print(f"  EXISTS  {filename}")
        return True

    session = get_session(token)
    progress = None
    tmp_path = os.path.join(os.getcwd(), filename + ".part")

    try:
        response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True,
                        leave=False, desc=filename[:40])

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                progress.update(len(chunk))

        progress.close()
        progress = None

        file_size = os.path.getsize(tmp_path)
        folder = destination_folder(filename, file_size)
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, filename)
        os.replace(tmp_path, dest)   # atomic on POSIX
        tmp_path = None
        print(f"  OK    {filename}  →  {folder}/")
        return True

    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        print(f"  FAIL  {filename}  HTTP {status}")
        return False
    except requests.Timeout:
        print(f"  FAIL  {filename}  timed out")
        return False
    except requests.RequestException as e:
        print(f"  FAIL  {filename}  {type(e).__name__}: {e}")
        return False
    except OSError as e:
        print(f"  FAIL  {filename}  filesystem error: {e}")
        return False
    finally:
        if progress is not None:
            progress.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def read_url_file(path: str) -> list[tuple[str, str]]:
    """Parse a file of  'filename - url'  lines.

    Skips blank lines and lines without the separator.
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or " - " not in line:
                continue
            filename, _, url = line.partition(" - ")
            filename = filename.strip()
            url = url.strip()
            if not filename or not url:
                print(f"  WARN  line {lineno}: malformed entry, skipping")
                continue
            pairs.append((filename, url))
    return pairs


def get_token() -> str:
    """Token priority: CIVITAI_API_TOKEN env var → interactive prompt."""
    token = os.environ.get("CIVITAI_API_TOKEN", "")
    if token:
        return token
    try:
        token = getpass.getpass("CivitAI API token (Enter to skip): ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)
    return token


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Download CivitAI files from a 'filename - url' list."
    )
    parser.add_argument("--url_file", required=True,
                        help="Text file with 'filename - url' lines")
    parser.add_argument("--max_threads", type=int, default=5,
                        help="Concurrent downloads (default: 5)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retry attempts for failed downloads (default: 3)")
    args = parser.parse_args()

    token = get_token()
    pairs = read_url_file(args.url_file)
    if not pairs:
        print("No valid entries found in URL file.")
        sys.exit(1)

    print(f"Found {len(pairs)} file(s) to download.\n")
    pending = pairs

    for attempt in range(1, args.retries + 1):
        if attempt > 1:
            print(f"\nRetry attempt {attempt}/{args.retries} "
                  f"({len(pending)} file(s) remaining)...\n")

        with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
            results = list(executor.map(
                lambda p: (p, download_one(p[0], p[1], token)),
                pending
            ))

        pending = [pair for pair, ok in results if not ok]
        if not pending:
            print("\nAll files downloaded successfully.")
            break
    else:
        print(f"\n{len(pending)} file(s) could not be downloaded after "
              f"{args.retries} attempts:")
        for filename, _ in pending:
            print(f"  {filename}")
        sys.exit(1)


if __name__ == "__main__":
    main()
