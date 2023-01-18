import contextlib
import hashlib
import itertools
import os
import os.path
import re
import sys
from typing import Any, Iterator, Optional, Tuple
from urllib.parse import urlparse
import warnings
import zipfile

import requests
from tqdm import tqdm

__all__ = ["download_dataset"]

USER_AGENT = "pytorch/vision"


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def download_url(
    url: str,
    root: str,
    filename: str,
    md5: Optional[str] = None,
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        # return download_file_from_google_drive(file_id, root, filename, md5)

        # Download a Google Drive file from  and place it in root.
        # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

        url = "https://drive.google.com/uc"
        params = dict(id=file_id, export="download")
        api_response = ""
        with requests.Session() as session:
            response = session.get(url, params=params, stream=True)

            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    token = value
                    break
            else:
                api_response, content = _extract_gdrive_api_response(response)
                token = "t" if api_response == "Virus scan warning" else None

            if token is not None:
                response = session.get(
                    url, params=dict(params, confirm=token), stream=True
                )
                api_response, content = _extract_gdrive_api_response(response)

            if api_response == "Quota exceeded":
                raise RuntimeError(
                    f"The daily quota of the file {filename} is exceeded and it "
                    f"can't be downloaded. This is a limitation of Google Drive "
                    f"and can only be overcome by trying again later."
                )

            _save_response_content(content, fpath)

        # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text
        if os.stat(fpath).st_size < 10 * 1024:
            with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:
                text = fh.read()
                # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
                if re.search(
                    r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)", text
                ):
                    warnings.warn(
                        f"We detected some HTML elements in the downloaded file. "
                        f"This most likely means that the download triggered an unhandled API response by GDrive. "
                        f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "
                        f"the response:\n\n{text}"
                    )

        if md5 and not check_md5(fpath, md5):
            raise RuntimeError(
                f"The MD5 checksum of the download file {fpath} does NOT MATCH the one on record."
                f"Please delete the file and try again. "
            )
        else:
            print(
                f"The MD5 checksum of the download file {fpath} does MATCH the one on record."
            )


def _extract_gdrive_api_response(
    response, chunk_size: int = 32 * 1024
) -> Tuple[bytes, Iterator[bytes]]:
    content = response.iter_content(chunk_size)
    first_chunk = None
    # filter out keep-alive new chunks
    while not first_chunk:
        first_chunk = next(content)
    content = itertools.chain([first_chunk], content)

    try:
        match = re.search(
            "<title>Google Drive - (?P<api_response>.+?)</title>", first_chunk.decode()
        )
        api_response = match["api_response"] if match is not None else None
    except UnicodeDecodeError:
        api_response = None
    return api_response, content


def _extract_zip(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(
        from_path,
        "r",
        compression=zipfile.ZIP_STORED,
    ) as zip:
        zip.extractall(to_path)


files = {
    20: {
        "url": "https://drive.google.com/file/d/12UU2yEf6nZ7HzLluALyFxZmPI5tN_iAb",
        "filename": "OmnImage84_20.zip",
        "md5": "30aa0b55fc6b3bccd06aaa6615661ee8",
    },
    100: {
        "url": "https://drive.google.com/file/d/1Ut_uITA2Y87Z6zHyGMMAbYBJd0MiNL8W",
        "filename": "OmnImage84_100.zip",
        "md5": "3869650152622568a7356146307c414e",
    },
    "sample": {
        "url": "https://drive.google.com/file/d/1TZoD4b48dgrX3cisKKlYJqBba0zn2V6v",
        "filename": "ImagenetSample.zip",
        "md5": "971eddceacb7e929cfbe55d041e9f794",
    },
}


def download_dataset(
    samples: int,
    download_root: str,
    extract_root: Optional[str] = None,
    remove_finished: bool = True,
) -> None:

    info = files[samples]
    url = info["url"]
    filename = info["filename"]
    md5 = info["md5"]

    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")

    _extract_zip(archive, extract_root)
    if remove_finished:
        os.remove(archive)
