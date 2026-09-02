import functools
import os
import socket
import time
from typing import Any
from urllib.parse import urlparse

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def ensure_no_proxy(host: str) -> None:
    parsed = urlparse(host)
    hostname = parsed.hostname or host
    hostname = hostname.strip("[]")
    for env_name in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()]
        if hostname not in entries:
            entries.append(hostname)
            os.environ[env_name] = ",".join(entries)


def port_open(host: str, port: int, timeout: float) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


class OpenPiPolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        api_key: str | None = None,
        open_timeout: float = 5.0,
    ) -> None:
        ensure_no_proxy(host)
        uri = host if host.startswith("ws") else f"ws://{host}"
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._packer = Packer()
        self._ws = websockets.sync.client.connect(
            f"{uri}:{port}",
            compression=None,
            max_size=None,
            open_timeout=open_timeout,
            additional_headers=headers,
        )
        self._metadata = unpackb(self._ws.recv())

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.infer_timed(observation)
        return result

    def infer_timed(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int | float]]:
        pack_start_wall_ns = time.time_ns()
        pack_start_mono_ns = time.monotonic_ns()
        payload = self._packer.pack(observation)
        pack_done_mono_ns = time.monotonic_ns()

        send_start_wall_ns = time.time_ns()
        send_start_mono_ns = time.monotonic_ns()
        self._ws.send(payload)
        send_done_wall_ns = time.time_ns()
        send_done_mono_ns = time.monotonic_ns()

        response = self._ws.recv()
        response_wall_ns = time.time_ns()
        response_mono_ns = time.monotonic_ns()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server error: {response}")
        result = unpackb(response)
        unpack_done_wall_ns = time.time_ns()
        unpack_done_mono_ns = time.monotonic_ns()
        timing = {
            "pack_start_wall_time_ns": pack_start_wall_ns,
            "request_send_start_wall_time_ns": send_start_wall_ns,
            "request_send_complete_wall_time_ns": send_done_wall_ns,
            "response_received_wall_time_ns": response_wall_ns,
            "response_unpacked_wall_time_ns": unpack_done_wall_ns,
            "pack_start_monotonic_time_ns": pack_start_mono_ns,
            "request_send_start_monotonic_time_ns": send_start_mono_ns,
            "request_send_complete_monotonic_time_ns": send_done_mono_ns,
            "response_received_monotonic_time_ns": response_mono_ns,
            "response_unpacked_monotonic_time_ns": unpack_done_mono_ns,
            "pack_ms": (pack_done_mono_ns - pack_start_mono_ns) / 1e6,
            "send_ms": (send_done_mono_ns - send_start_mono_ns) / 1e6,
            "wait_response_ms": (response_mono_ns - send_done_mono_ns) / 1e6,
            "unpack_ms": (unpack_done_mono_ns - response_mono_ns) / 1e6,
            "total_ms": (unpack_done_mono_ns - pack_start_mono_ns) / 1e6,
        }
        return result, timing

    def close(self) -> None:
        self._ws.close()
