"""Bypass the cluster's broken internal DNS for huggingface.co.

The cluster's default resolver (192.168.101.5/.6) returns SERVFAIL for
``huggingface.co`` and ``hf-mirror.com`` but general internet (google.com) is
fine. 8.8.8.8 / 1.1.1.1 return correct HF CloudFront IPs.

We cannot edit /etc/resolv.conf (no root) and cannot ``pip install dnspython``
(PyPI DNS also blocked). So: monkey-patch ``socket.getaddrinfo`` with a pure-
stdlib DNS-over-UDP lookup against a list of known-good public resolvers,
cached in-process.

Import this module BEFORE any HuggingFace-facing code (``datasets``,
``huggingface_hub``) in any script that fetches from HF.

Usage:
    from visword.hf_dns_shim import install
    install()
"""
from __future__ import annotations

import os
import socket
import struct
from typing import Iterable


# Hosts to override. Anything not in this set falls through to the OS resolver.
_HF_HOSTS = frozenset({
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-eu-1.huggingface.co",
    "hf-mirror.com",
    "files.pythonhosted.org",   # pip blocked too; handy if we need to install
    "pypi.org",
})

# Resolvers that do respond correctly to huggingface.co from this cluster.
_PUBLIC_DNS = ("8.8.8.8", "1.1.1.1", "8.8.4.4", "9.9.9.9")

# Per-process cache: hostname -> list[str] of IPv4 strings.
_cache: dict[str, list[str]] = {}


def _build_query(host: str, qid: int = 0x1234) -> bytes:
    flags = 0x0100  # standard query, recursion desired
    header = struct.pack(">HHHHHH", qid, flags, 1, 0, 0, 0)
    qname = b""
    for label in host.rstrip(".").split("."):
        qname += bytes([len(label)]) + label.encode("ascii")
    qname += b"\x00"
    qtype_class = struct.pack(">HH", 1, 1)  # A, IN
    return header + qname + qtype_class


def _parse_answers(data: bytes) -> list[str]:
    if len(data) < 12:
        return []
    _, flags, qd, an, _, _ = struct.unpack(">HHHHHH", data[:12])
    rcode = flags & 0x0F
    if rcode != 0 or an == 0:
        return []

    # Skip question section.
    off = 12
    for _ in range(qd):
        while off < len(data) and data[off] != 0:
            off += data[off] + 1
        off += 1 + 4   # null byte + QTYPE + QCLASS

    ips: list[str] = []
    for _ in range(an):
        # NAME: either pointer (2 bytes) or labels (variable)
        if off >= len(data):
            break
        if data[off] & 0xC0:
            off += 2
        else:
            while off < len(data) and data[off] != 0:
                off += data[off] + 1
            off += 1
        if off + 10 > len(data):
            break
        rtype, _, _, rdlen = struct.unpack(">HHIH", data[off:off + 10])
        off += 10
        if rtype == 1 and rdlen == 4 and off + 4 <= len(data):
            ips.append(".".join(str(b) for b in data[off:off + 4]))
        off += rdlen
    return ips


def _resolve_public(host: str, timeout: float = 3.0) -> list[str]:
    if host in _cache:
        return _cache[host]
    query = _build_query(host)
    for resolver in _PUBLIC_DNS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            sock.sendto(query, (resolver, 53))
            data, _ = sock.recvfrom(512)
            ips = _parse_answers(data)
            if ips:
                _cache[host] = ips
                return ips
        except (OSError, socket.timeout):
            continue
        finally:
            sock.close()
    return []


_orig_getaddrinfo = socket.getaddrinfo

# Hosts that are KNOWN to be SERVFAIL'd by the internal cluster resolver.
# For these, try 8.8.8.8 FIRST without ever calling the OS resolver.
_FORCE_PUBLIC = frozenset({
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-eu-1.huggingface.co",
    "hf-mirror.com",
    "files.pythonhosted.org",
    "pypi.org",
    # Wildcards for any *.huggingface.co subdomain are handled in code below.
})


def _is_hf_related(host: str) -> bool:
    return (
        host in _FORCE_PUBLIC
        or host.endswith(".huggingface.co")
        or host.endswith(".hf.co")
        or host.endswith(".pypi.org")
        or host.endswith(".pythonhosted.org")
    )


def _patched_getaddrinfo(host, port, *args, **kwargs):
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if _is_hf_related(host):
        ips = _resolve_public(host)
        if not ips:
            # Subdomains without A records (e.g. cdn-lfs is a CNAME) fall
            # back to the apex IPs — CloudFront wildcard cert routes by SNI.
            if host.endswith(".huggingface.co"):
                ips = _resolve_public("huggingface.co")
        if ips:
            port_int = port if isinstance(port, int) else int(port) if port else 443
            return [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                 (ip, port_int))
                for ip in ips
            ]
    return _orig_getaddrinfo(host, port, *args, **kwargs)


def install() -> None:
    """Idempotent: call multiple times, only patches once."""
    if getattr(socket.getaddrinfo, "_hf_dns_shimmed", False):
        return
    _patched_getaddrinfo._hf_dns_shimmed = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _patched_getaddrinfo


def test_resolution() -> dict[str, list[str]]:
    """Utility: return resolved IPs for every host in the override set."""
    return {h: _resolve_public(h) for h in _HF_HOSTS}


if __name__ == "__main__":
    # Self-test when run as a script.
    install()
    for host in sorted(_HF_HOSTS):
        ips = _resolve_public(host)
        status = ", ".join(ips) if ips else "UNRESOLVED"
        print(f"{host:40s} -> {status}")
