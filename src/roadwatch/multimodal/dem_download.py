"""Download and unpack the NYC 1-foot DEM (dpc8-z3jc, 2010 LiDAR) into data/raw/multimodal/dem/ with a progress log.

The archive is a single Erdas Imagine mosaic of the five boroughs (~18 GB compressed, ~99 GB unpacked). It cannot be read in
windows over HTTP (deflate), so it is fetched once, checksummed and kept; Brooklyn is then read with windowed raster access.
Usage: python -I -X utf8 -m roadwatch.multimodal.dem_download
"""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path

from remotezip import RemoteZip

from ..config import Paths

URL = "https://sa-static-customer-assets-us-east-1-fedramp-prod.s3.amazonaws.com/data.cityofnewyork.us/NYC_DEM_1ft_Float_2.zip"


def main() -> None:
    out = Paths().raw / "multimodal" / "dem"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "download.log"
    t0 = time.time()
    with RemoteZip(URL) as z:
        infos = z.infolist()
        for i in infos:
            dest = out / i.filename
            if dest.exists() and dest.stat().st_size == i.file_size:
                continue
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} start {i.filename} compressed={i.compress_size} size={i.file_size}\n")
            with z.open(i) as src, open(dest, "wb") as dst:
                n, h = 0, hashlib.sha256()
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
                    n += len(chunk)
                    if (n // (8 * 1024 * 1024)) % 64 == 0:
                        with log.open("a", encoding="utf-8") as f:
                            f.write(f"{time.strftime('%H:%M:%S')} {i.filename} {n / 1e9:.1f} GB\n")
        man = {"source": URL, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "files": [{"name": i.filename, "size": (out / i.filename).stat().st_size, "crc32": f"{i.CRC:08x}"} for i in infos],
               "seconds": round(time.time() - t0)}
        (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} DONE\n")


if __name__ == "__main__":
    main()
