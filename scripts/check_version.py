#!/usr/bin/env python3
#
#  check_version.py
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  The version is written three times: the macros in include/snicholls/formats/version.hpp (and its string), the
#  project(... VERSION) in CMakeLists.txt, and the release tag. They are compared on every build:
#
#      python3 scripts/check_version.py                 # header vs CMake
#      python3 scripts/check_version.py --tag v0.1.0    # ... and vs a tag
#
#  Exit status 0 when they agree, 1 when they do not.
#

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "include", "snicholls", "formats", "version.hpp")
CMAKE = os.path.join(ROOT, "CMakeLists.txt")


def header_version():
    src = open(HEADER, encoding="utf-8").read()
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(r"^#define\s+SNICHOLLS_FORMATS_VERSION_%s\s+(\d+)\s*$" % field, src, re.M)
        if not m:
            raise SystemExit("check_version: SNICHOLLS_FORMATS_VERSION_%s not found" % field)
        parts.append(m.group(1))
    m = re.search(r'string\s*=\s*"([0-9.]+)"', src)
    if not m:
        raise SystemExit("check_version: version::string not found")
    return ".".join(parts), m.group(1)


def cmake_version():
    src = open(CMAKE, encoding="utf-8").read()
    m = re.search(r"project\s*\(\s*snicholls_formats\s+VERSION\s+(\d+\.\d+\.\d+)", src, re.S)
    if not m:
        raise SystemExit("check_version: project(snicholls_formats VERSION x.y.z) not found in CMakeLists.txt")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser(description="Check the version is the same everywhere")
    ap.add_argument("--tag", default=os.environ.get("RELEASE_TAG"), help="a git tag to check too")
    args = ap.parse_args()
    hdr, text = header_version()
    cml = cmake_version()
    print("  include/snicholls/formats/version.hpp   %s (string %s)" % (hdr, text))
    print("  CMakeLists.txt                          %s" % cml)
    good = hdr == cml == text
    if args.tag:
        print("  git tag                                 %s" % args.tag)
        good = good and args.tag.lstrip("v") == hdr
    if not good:
        print("check_version: FAILED - one release, one number", file=sys.stderr)
        return 1
    print("version check: consistent (%s)" % hdr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
