#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download and install the MakeHuman system assets.

**Licensing:**         AGPL3 (this script); the downloaded assets are CC0.

    This file is part of MakeHuman Community (www.makehumancommunity.org).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Abstract
--------

The source tree only ships the base mesh, the targets and a few defaults.
Hair, eyebrows, eyelashes, clothes, skins, proxies, teeth, tongue and the
extra eye materials used to come from download_assets_git.py, which clones
github.com/makehumancommunity/makehuman-assets. That repository no longer
exists. The same CC0 assets are published by the MakeHuman Community as the
"makehuman_system_assets" asset pack:

    https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html

This script downloads that pack, verifies its SHA-256 and extracts it into
the data folder of this source checkout. Files that already exist are never
overwritten.
"""

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

PACK_URLS = (
    # files. is fast; files2. is a mirror that measured ~100x slower (2026-11).
    "https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip",
    "https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip",
)
# Pack published 2024-04-14 (Last-Modified), 280737770 bytes.
PACK_SHA256 = "b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107"

CHUNK = 1024 * 1024
# Written into the data folder after a complete install; holds the pack's SHA-256
MARKER = ".system_assets_installed"


class AssetError(Exception):
    """Raised when the asset pack cannot be downloaded, verified or installed."""


def default_data_dir():
    """The data folder of the source checkout this script lives in."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(urls, dest, timeout=60, log=print):
    """Download the first URL that works into dest. Returns the URL used."""
    errors = []
    for url in urls:
        log("Downloading %s" % url)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, open(dest, "wb") as out:
                shutil.copyfileobj(response, out, CHUNK)
            return url
        except OSError as e:  # URLError, HTTPError, timeouts and disk errors are all OSError
            log("  failed: %s" % e)
            errors.append("%s: %s" % (url, e))
    raise AssetError("Could not download the asset pack from any mirror:\n  " + "\n  ".join(errors))


def check_member(info):
    """Reject zip entries that could write outside the target folder."""
    name = info.filename
    parts = name.replace("\\", "/").split("/")
    if name.startswith(("/", "\\")) or ".." in parts or (len(name) > 1 and name[1] == ":"):
        raise AssetError("Unsafe path in asset pack: %r" % name)
    if stat.S_ISLNK(info.external_attr >> 16):
        raise AssetError("Symbolic link in asset pack: %r" % name)


def install(zip_path, data_dir, log=print):
    """
    Extract the pack into data_dir without overwriting anything.
    Returns a dict with the counts of 'installed', 'identical' and 'kept'
    (existing files whose content differs from the pack).
    """
    counts = {"installed": 0, "identical": 0, "kept": 0}
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        for info in members:
            check_member(info)
        for info in members:
            if info.is_dir():
                continue
            target = os.path.join(data_dir, *info.filename.split("/"))
            if os.path.exists(target):
                with open(target, "rb") as f:
                    same = f.read() == z.read(info)
                counts["identical" if same else "kept"] += 1
                if not same:
                    log("Keeping existing file %s" % target)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            counts["installed"] += 1
    return counts


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("Abstract")[0].strip().splitlines()[0])
    parser.add_argument("--zip", help="use an already downloaded makehuman_system_assets_cc0.zip")
    parser.add_argument("--data-dir", default=default_data_dir(),
                        help="where to install (default: the data folder next to this script)")
    parser.add_argument("--sha256", default=PACK_SHA256,
                        help="expected SHA-256 of the pack (only change it if the pack was republished)")
    parser.add_argument("--force", action="store_true",
                        help="download and install again even if this pack is already installed")
    return parser.parse_args(argv)


def installed_pack(data_dir):
    """SHA-256 recorded by the last complete install into data_dir, or None."""
    try:
        with open(os.path.join(data_dir, MARKER), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def main(argv=None, urls=PACK_URLS, log=print):
    args = parse_args(argv)
    if not os.path.isdir(args.data_dir):
        log("Data folder not found: %s" % args.data_dir)
        return 2
    if not args.force and installed_pack(args.data_dir) == args.sha256.lower():
        log("The system assets are already installed in %s (use --force to reinstall)." % args.data_dir)
        return 0

    tmpdir = None
    try:
        if args.zip:
            zip_path = args.zip
        else:
            tmpdir = tempfile.mkdtemp(prefix="mh-assets-")
            zip_path = os.path.join(tmpdir, "makehuman_system_assets_cc0.zip")
            download(urls, zip_path, log=log)

        actual = sha256_of(zip_path)
        if actual != args.sha256.lower():
            log("SHA-256 mismatch for %s\n  expected %s\n  got      %s" % (zip_path, args.sha256, actual))
            log("If the pack was republished on purpose, re-run with --sha256 %s" % actual)
            return 3

        counts = install(zip_path, args.data_dir, log=log)
        with open(os.path.join(args.data_dir, MARKER), "w", encoding="utf-8") as f:
            f.write(actual + "\n")
    except AssetError as e:
        log(str(e))
        return 1
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    log("Installed %(installed)d files (%(identical)d already present, %(kept)d existing files kept)." % counts)
    log("Run compile_models.py and compile_proxies.py afterwards for faster loading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
