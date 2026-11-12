"""Which CycloneDDS config the standalone launcher uses (packaging/quadsim -> quadsim.launch).

First match wins:
  1. --dds-config <path>    any XML file; `--dds-config none` leaves CYCLONEDDS_URI unset (Cyclone defaults:
                            multicast on every interface)
  2. CYCLONEDDS_URI         already set by the caller: passed through as is (a file:// URI or inline XML)
  3. CYCLONEDDS_CONFIG      a file name under the bundled conf/ (the same knob as docker/run.sh)
  4. conf/cyclonedds.xml    the bundled default (loopback only, Domain 1, shared with WebRvizLite)

Cyclone applies a <Domain Id="N"> block only to domain N and silently falls back to its defaults otherwise, so a
config whose Domain Id differs from ROS_DOMAIN_ID (default 1, as in docker/run.sh) gets a warning.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_FILE = "cyclonedds.xml"
DEFAULT_DOMAIN = 1
FLAG = "--dds-config"
NONE = "none"
_DOMAIN_ID = re.compile(r"<Domain\b[^>]*\bId\s*=\s*[\"']([^\"']*)[\"']")


class DdsConfigError(ValueError):
    """A DDS setting from the command line or the environment that cannot be used."""


@dataclass(frozen=True)
class DdsChoice:
    uri: str | None                 # CYCLONEDDS_URI to export; None = leave it unset
    source: str                     # which rule picked it, printed by the launcher
    domain_id: int                  # ROS_DOMAIN_ID to export
    warnings: tuple[str, ...] = ()


def resolve(argv: list[str], env: dict[str, str], conf_dir) -> tuple[DdsChoice, list[str]]:
    """Pick the DDS config from argv (the --dds-config flag is consumed) and env; return it and the other args."""
    flag, rest = _take_flag(argv)
    domain = _domain_id(env)
    uri, source = _pick(flag, env, str(conf_dir))
    return DdsChoice(uri, source, domain, _domain_warnings(uri, domain)), rest


def _take_flag(argv: list[str]) -> tuple[str | None, list[str]]:
    value, rest, it = None, [], iter(argv)
    for arg in it:
        if arg == FLAG:
            value = next(it, None)
            if value is None:
                raise DdsConfigError(f"{FLAG} needs a value: an XML path or '{NONE}'")
        elif arg.startswith(FLAG + "="):
            value = arg[len(FLAG) + 1:]
        else:
            rest.append(arg)
    return value, rest


def _domain_id(env: dict[str, str]) -> int:
    raw = env.get("ROS_DOMAIN_ID", "").strip() or str(DEFAULT_DOMAIN)
    if not raw.isdigit():
        raise DdsConfigError(f"ROS_DOMAIN_ID={raw!r}: expected a non-negative integer")
    return int(raw)


def _pick(flag: str | None, env: dict[str, str], conf_dir: str) -> tuple[str | None, str]:
    if flag == NONE:
        return None, f"{FLAG} {NONE}"
    if flag is not None:
        return _file_uri(os.path.abspath(flag), FLAG), FLAG
    if env.get("CYCLONEDDS_URI"):
        return env["CYCLONEDDS_URI"], "CYCLONEDDS_URI"
    if env.get("CYCLONEDDS_CONFIG"):
        return _file_uri(os.path.join(conf_dir, env["CYCLONEDDS_CONFIG"]), "CYCLONEDDS_CONFIG"), "CYCLONEDDS_CONFIG"
    return _file_uri(os.path.join(conf_dir, DEFAULT_FILE), "the bundled default"), "default"


def _file_uri(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise DdsConfigError(f"{what}: no such file {path}")
    return "file://" + os.path.normpath(path)


def _domain_warnings(uri: str | None, domain: int) -> tuple[str, ...]:
    xml = _xml_text(uri)
    ids = {i for i in _DOMAIN_ID.findall(xml) if i.lower() != "any"}
    if not ids or str(domain) in ids:
        return ()
    shown = ",".join(sorted(ids))
    return (f"DDS config has Domain Id={shown} but ROS_DOMAIN_ID={domain}: Cyclone ignores that block and falls "
            f"back to its defaults (multicast); set ROS_DOMAIN_ID={shown} or use a config for domain {domain}",)


def _xml_text(uri: str | None) -> str:
    if uri is None:
        return ""
    if uri.lstrip().startswith("<"):
        return uri
    if uri.startswith("file://") and "," not in uri:
        with open(uri[len("file://"):], encoding="utf-8") as f:
            return f.read()
    return ""                       # a URI list or another scheme: not checked
