"""Runs ArcFaceEmbedder in a separate, spawned process.

mediapipe and onnxruntime each embed their own protobuf-generated code. When
both are loaded in the same process, mediapipe's frozen (statically linked)
protobuf descriptors and onnxruntime's dynamically-linked protobuf runtime
disagree on internal layout, and the process aborts or segfaults the moment
onnxruntime registers its own descriptors — regardless of which system
protobuf version is installed (tested against two independently mutually-
matched combinations; both crashed).

`FaceProcessor` already imports mediapipe's `Landmarker` in the main process,
so the embedder runs in a child process that never imports mediapipe. The
child is started with the "spawn" method rather than the default "fork": fork
would copy the parent's address space, mediapipe's already-loaded library and
all, straight into the child, defeating the isolation.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np

EMBEDDING_DIMENSIONS = 512


def _worker_main(model_path: str, providers: Optional[list], conn) -> None:
    try:
        from .embed import ArcFaceEmbedder

        embedder = ArcFaceEmbedder(Path(model_path), providers=providers)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised here
        conn.send(("error", repr(exc)))
        return

    conn.send(("ready", None))

    while True:
        try:
            crop = conn.recv()
        except EOFError:
            break
        if crop is None:
            break
        try:
            embedding = embedder.embed(crop)
            conn.send(("ok", embedding))
        except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised here
            conn.send(("error", repr(exc)))


class IsolatedArcFaceEmbedder:
    """Drop-in replacement for `ArcFaceEmbedder`: same `embed()` signature,
    but the model runs in a spawned child process so its onnxruntime never
    shares an address space with mediapipe's."""

    def __init__(
        self, model_path: Optional[Path] = None, providers: Optional[list[str]] = None
    ) -> None:
        from . import paths

        self.model_path = Path(model_path or paths.arcface_model())
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {self.model_path}. "
                "Fetch it with `glancectl fetch-model`."
            )

        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_worker_main,
            args=(str(self.model_path), providers, child_conn),
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # only the child needs its end

        status, payload = self._parent_conn.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"embedding worker failed to start: {payload}")

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        if aligned_rgb.shape[:2] != (112, 112):
            raise ValueError(f"expected a 112x112 crop, got {aligned_rgb.shape[:2]}")
        if not self._process.is_alive():
            raise RuntimeError("embedding worker is not running")
        self._parent_conn.send(aligned_rgb)
        status, payload = self._parent_conn.recv()
        if status != "ok":
            raise RuntimeError(f"embedding worker error: {payload}")
        return payload

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._parent_conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
        self._parent_conn.close()
