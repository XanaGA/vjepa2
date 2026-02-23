import os
import requests
import shutil
import argparse
from tqdm import tqdm


def download_single_category(link_file: str,
                             download_folder: str,
                             category_name: str):
    """
    Download and extract a single CO3D category from a TSV link file.

    Expected file format:
        file_name<TAB>cdn_link

    Example row:
        CO3D_car.zip    https://...
    """

    if not os.path.isfile(link_file):
        raise ValueError("Invalid link file path.")

    os.makedirs(download_folder, exist_ok=True)

    target_filename = f"CO3D_{category_name}.zip"
    found = False

    with open(link_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            file_name, url = line.split("\t")

            if file_name == target_filename:
                found = True
                local_path = os.path.join(download_folder, file_name)

                print(f"\nDownloading {file_name}...")
                response = requests.get(url, stream=True)
                total = int(response.headers.get("content-length", 0))

                with open(local_path, "wb") as out_file, tqdm(
                    desc=file_name,
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            size = out_file.write(chunk)
                            bar.update(size)

                print(f"Extracting {file_name}...")
                shutil.unpack_archive(local_path, download_folder)

                print("Done.")
                break

    if not found:
        raise ValueError(f"Category '{category_name}' not found in link file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a single CO3D category from TSV links file."
    )

    parser.add_argument(
        "--link_file",
        type=str,
        required=True,
        help="Path to TSV file containing download links."
    )

    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="Target directory for downloads."
    )

    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category name (e.g. car, chair, orange)."
    )

    args = parser.parse_args()

    download_single_category(
        link_file=args.link_file,
        download_folder=args.download_folder,
        category_name=args.category,
    )
