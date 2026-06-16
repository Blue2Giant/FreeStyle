---
pretty_name: SREF OneIG Dataset
---

# SREF OneIG Dataset

This folder contains the SREF OneIG dataset exported from the VGO data configuration.  The image payload is distributed as **tar shards** instead of millions of individual image files, which makes upload/download much more reliable on Hugging Face.

## Directory layout

```text
sref/
  README.md
  export_summary.json
  pairs.csv
  metadata.jsonl
  shards_manifest.jsonl
  shards_summary.json
  image_tar_shards/
    images_00.tar
    images_01.tar
    ...
    images_ff.tar
```

Important files:

- `pairs.csv`: pair-level records. Image paths are relative paths such as `images/00/<uuid>.jpg`.
- `metadata.jsonl`: sequence-level metadata.
- `export_summary.json`: export statistics.
- `shards_manifest.jsonl`: one JSON record per tar shard, including prefix, file count, and byte size.
- `shards_summary.json`: global shard summary.
- `image_tar_shards/images_*.tar`: uncompressed image tar shards.

The tar shards preserve the original image paths. For example, `images_0a.tar` contains files like:

```text
images/0a/<image_id>.jpg
```

## Dataset scale

Current shard export summary:

```text
Image files: 1,238,604
Tar shards: 256
Shard payload size: about 488 GiB
```

The metadata files are large as well:

```text
pairs.csv      ~3.6 GiB
metadata.jsonl ~2.9 GiB
```

## Download

Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub hf_xet
```

Download the full `sref/` folder:

```bash
huggingface-cli download Blue2Giant/FreeStyle_Dataset \
  --repo-type dataset \
  --include 'sref/*' 'sref/image_tar_shards/*.tar' \
  --local-dir ./FreeStyle_Dataset
```

If you only want metadata first:

```bash
huggingface-cli download Blue2Giant/FreeStyle_Dataset \
  --repo-type dataset \
  --include 'sref/README.md' 'sref/export_summary.json' 'sref/pairs.csv' 'sref/metadata.jsonl' 'sref/shards_*.json*' \
  --local-dir ./FreeStyle_Dataset
```

## Extract images

After download, extract the image shards at the `sref/` folder root:

```bash
cd ./FreeStyle_Dataset/sref
for f in image_tar_shards/images_*.tar; do
  tar -xf "$f"
done
```

After extraction, the expected layout is:

```text
sref/
  pairs.csv
  metadata.jsonl
  images/
    00/
    01/
    ...
    ff/
```

Image references in `pairs.csv` and `metadata.jsonl` should then resolve relative to the `sref/` directory.

## Quick verification

```bash
cd ./FreeStyle_Dataset/sref
python3 - <<'PY'
import json
from pathlib import Path
summary = json.load(open('shards_summary.json'))
print(summary)
print('num shard files:', len(list(Path('image_tar_shards').glob('images_*.tar'))))
print('pairs exists:', Path('pairs.csv').exists())
print('metadata exists:', Path('metadata.jsonl').exists())
PY
```

After extraction:

```bash
find images -type f | wc -l
# Expected: 1238604
```

## Python usage example

```python
from pathlib import Path
import csv

root = Path('./FreeStyle_Dataset/sref')
with open(root / 'pairs.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    print(row)
    # If a column contains a relative image path such as images/00/xxx.jpg:
    # image_path = root / row['<image_path_column>']
```

Column names may depend on the exporter version; inspect the first row/header of `pairs.csv` before writing downstream logic.

## Notes

- Tar files are intentionally uncompressed because the payload is JPEG and already compressed.
- Sharding is by the first two hex characters of the image id/path (`00` to `ff`).
- The shard layout is preferred over millions of individual remote files for reliability and speed.
