"""Create a Lambda layer zip from /opt/python and /opt/lib, preserving symlinks.

Run inside the Lambda base image container after building the layer:
    python make_layer_zip.py /opt /tmp/rag-layer-light.zip
"""
import os
import sys
import zipfile

SRC = sys.argv[1] if len(sys.argv) > 1 else "/opt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/rag-layer-light.zip"


def add_dir(zf, base, arc_base):
    for root, dirs, files in os.walk(base):
        for name in files:
            full = os.path.join(root, name)
            arc = os.path.join(arc_base, os.path.relpath(full, base))
            if os.path.islink(full):
                # Store as a symlink entry (external_attr marks it as a symlink).
                info = zipfile.ZipInfo(arc)
                info.create_system = 3  # Unix
                info.external_attr = 0o120777 << 16  # symlink mode
                zf.writestr(info, os.readlink(full))
            else:
                zf.write(full, arc)


with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    add_dir(zf, os.path.join(SRC, "python"), "python")
    add_dir(zf, os.path.join(SRC, "lib"), "lib")

print("Wrote", OUT)
