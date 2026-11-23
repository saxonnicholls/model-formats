#!/usr/bin/env python3
#
#  check_mirror.py
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  tests/ and demos/ mirror include/snicholls/formats/ (CLAUDE.md "Tests", "Demos"). For every public header
#  <sub>/<name>.hpp (detail/ and the umbrella formats.hpp excluded):
#
#    tests/<sub>/<name>_test.cpp    registered with model_formats_add_test(<sub>/<name> ...) in tests/CMakeLists.txt
#    demos/<sub>/<name>_demo.cpp    registered with model_formats_add_demo(<sub>/<name> ...) in demos/CMakeLists.txt
#
#  and every test or demo has its header. Exit status 1 on any finding.
#

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = os.path.join(ROOT, "include", "snicholls", "formats")
UMBRELLA = "formats.hpp"   # the umbrella: no test or demo of its own, it is compiled as the single include


def registered(tree, function):
    text = open(os.path.join(ROOT, tree, "CMakeLists.txt"), encoding="utf-8").read()
    return set(re.findall(r"^\s*%s\(\s*([\w/]+)\s" % function, text, re.M))


def files(tree, suffix):
    out = set()
    for dirpath, _, names in os.walk(os.path.join(ROOT, tree)):
        for n in names:
            if n.endswith(suffix):
                out.add(os.path.relpath(os.path.join(dirpath, n), os.path.join(ROOT, tree))[: -len(suffix)])
    return out


def main():
    headers = set()
    for dirpath, _, names in os.walk(INCLUDE):
        for n in names:
            rel = os.path.relpath(os.path.join(dirpath, n), INCLUDE)
            if n.endswith(".hpp") and not rel.startswith("detail" + os.sep) and rel != UMBRELLA:
                headers.add(rel[:-4])
    problems = []
    for tree, suffix, function in (("tests", "_test.cpp", "model_formats_add_test"), ("demos", "_demo.cpp", "model_formats_add_demo")):
        have, listed = files(tree, suffix), registered(tree, function)
        for h in sorted(headers):
            if h not in have:
                problems.append("%s.hpp: no %s/%s%s" % (h, tree, h, suffix))
            elif h not in listed:
                problems.append("%s/%s%s is not registered in %s/CMakeLists.txt" % (tree, h, suffix, tree))
        for f in sorted((have | listed) - headers):
            problems.append("%s/%s%s has no header include/snicholls/formats/%s.hpp" % (tree, f, suffix, f))
    for p in problems:
        print("check_mirror: " + p, file=sys.stderr)
    if problems:
        return 1
    print("mirror check: %d headers, each with its test and demo" % len(headers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
