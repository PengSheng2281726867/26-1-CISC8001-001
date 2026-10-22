#!/usr/bin/env bash
# Old distutils emits Sun-style -R for standalone CPython's generic "cc".
# Translate only that rpath spelling; preserve all compilation/link options.
set -euo pipefail
lift_link_args=()
for lift_link_arg in "$@"; do
    case "$lift_link_arg" in
        -R*) lift_link_args+=("-Wl,-rpath,${lift_link_arg#-R}") ;;
        *) lift_link_args+=("$lift_link_arg") ;;
    esac
done
exec gcc "${lift_link_args[@]}"
