"""
Content-hash cache: if the exact same image bytes come through twice (a real
scenario -- WhatsApp resends, clients re-uploading the same photo), we should
never pay for the same extraction twice. Keyed on sha256 of the resized image
bytes actually sent to the model, so a cache hit means "we would have sent
literally the same request."
"""

import hashlib
import json
import os
import tempfile


class ExtractionCache:
    def __init__(self, path: str):
        self.path = path
        self._store = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # A previous run could have been killed mid-save, leaving a
                # partial/corrupt cache file. That shouldn't crash this run
                # before any pipeline logic even executes -- start fresh
                # instead (worst case: a few avoidable cache misses).
                print(f"[cache] warning: could not read existing cache at {path} ({e}); starting fresh")
                self._store = {}

    @staticmethod
    def key_for(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image_bytes: bytes):
        return self._store.get(self.key_for(image_bytes))

    def set(self, image_bytes: bytes, record: dict):
        self._store[self.key_for(image_bytes)] = record

    def save(self):
        """
        Atomic write: build the file fully in a temp file in the same
        directory, then os.replace() it over the destination. Rename is
        atomic on POSIX, so a crash or kill signal mid-save can never leave
        a half-written, unparseable cache file for the next run to choke on.
        """
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".cache_tmp_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._store, f, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
