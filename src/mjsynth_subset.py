import argparse
import http.client
import os
import tarfile
import time
import urllib.request

import cv2
import numpy as np

from datasets import load_config


class ResumingStream:
    """A download that survives dropped connections and restarts of this script.

    Every byte is also kept in a partial file, so a rerun continues where the last one stopped.
    """

    def __init__(self, url, partial, timeout=60):
        self.url, self.timeout, self.response, self.total = url, timeout, None, None
        self.saved = os.path.getsize(partial) if os.path.exists(partial) else 0
        self.cache = open(partial, "a+b")
        self.cache.seek(0)
        self.pos, self.next_report = 0, 50_000_000

    def connect(self):
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-"})
        self.response = urllib.request.urlopen(request, timeout=self.timeout)
        self.total = int(self.response.headers["Content-Range"].rsplit("/", 1)[1])

    def from_network(self, size):
        for attempt in range(1000):
            try:
                if self.response is None:
                    self.connect()
                data = self.response.read(size)
                # a server that hangs up early looks like the end of the file
                if data or self.pos >= self.total:
                    return data
                raise ConnectionError("the server closed the connection early")
            except (OSError, http.client.HTTPException) as e:
                print(f"connection lost at {self.pos / 1e6:.0f} MB ({e}); reconnecting", flush=True)
                self.response = None
                time.sleep(min(60, 2 ** min(attempt, 6)))
        raise RuntimeError("the download kept failing")

    def read(self, size=-1):
        if self.pos < self.saved:
            self.cache.seek(self.pos)
            data = self.cache.read(min(size, self.saved - self.pos))
        else:
            data = self.from_network(size)
            self.cache.seek(0, os.SEEK_END)
            self.cache.write(data)
            self.cache.flush()
            self.saved += len(data)
        self.pos += len(data)
        if self.pos >= self.next_report:
            print(f"{self.pos / 1e6:.0f} MB read", flush=True)
            self.next_report += 50_000_000
        return data


def word_of(name):
    return os.path.basename(name).split("_")[1].lower()


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="stream the MJSynth tar.gz and keep only a subset of the images")
    ap.add_argument("--url", required=True, help="link to mjsynth.tar.gz from the VGG text page")
    ap.add_argument("--out", default=cfg["public"]["mjsynth"]["root"])
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth image; use about 45 if the archive is ordered by word (this reads all of it)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines, broken, seen = [], 0, 0
    # "r|gz" reads the archive as a stream, so the download stops as soon as we break out
    partial = os.path.join(os.path.dirname(os.path.normpath(args.out)), "mjsynth_partial.tar.gz")
    with tarfile.open(fileobj=ResumingStream(args.url, partial), mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue
            seen += 1
            if (seen - 1) % args.every:
                continue
            data = tar.extractfile(member).read()
            # a few MJSynth images are known to be unreadable, so each one is decoded once here
            if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) is None:
                broken += 1
                continue
            rel = member.name.split("90kDICT32px/")[-1]
            path = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            lines.append(f"./{rel} {len(lines)}")
            if len(lines) % 10000 == 0:
                print(f"{len(lines)} images kept, {seen} read")
            if len(lines) >= args.n:
                break

    os.remove(partial)
    # same format as the original annotation file, so load_mjsynth reads it unchanged
    with open(os.path.join(args.out, cfg["public"]["mjsynth"]["train"]), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    words = {word_of(line.split()[0]) for line in lines}
    print(f"kept {len(lines)} images ({len(words)} unique words) in {args.out}, skipped {broken} unreadable ones")
    # MJSynth has ~90k words over ~9M images, so a well-mixed 200k sample has tens of thousands of words
    if lines and len(words) < 0.1 * len(lines):
        print("few unique words: the archive looks ordered by word, rerun with --every 45")


if __name__ == "__main__":
    main()
