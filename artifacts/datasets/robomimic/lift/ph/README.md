# robomimic Lift PH image and low-dimensional data

The original HDF5 datasets are intentionally not stored in Git because they are large. Place downloaded files in this directory; the repository tracks this note and the small JSON manifests only. The generated `visualizations/` files are local previews and are not required for training.

## Files

| File | Size | SHA-256 | Contents |
|---|---:|---|---|
| `low_dim.hdf5` | 18,548,952 bytes | `18106c8c072020672fecdd2a9ffbfe7dc656823cb877f972ade7fad34d2a5b74` | 200 demos; 9,666 frames; low-dimensional observations |
| `image.hdf5` | 837,217,792 bytes | `87f813fa5118eaaac520f9933dc34e03336e307929c5f1f5fc53fb9609e1789e` | 200 demos; 9,666 frames; two 84×84 RGB camera streams plus low-dimensional observations |

The image cameras are `agentview_image` and `robot0_eye_in_hand_image`. Both files are robomimic 0.2.0 Lift / PH datasets. Their verified metadata and download sources are recorded in `download_manifest.json` and `image_download_manifest.json`.

## Download

From the repository root, download the low-dimensional file with:

```powershell
python -B -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --download_dir artifacts/datasets/robomimic
```

For the image file, use the official source URL recorded in `image_download_manifest.json`:

```powershell
New-Item -ItemType Directory -Force artifacts/datasets/robomimic/lift/ph | Out-Null
curl.exe -L --fail --retry 20 --retry-delay 2 --retry-all-errors --continue-at - --output artifacts/datasets/robomimic/lift/ph/image.hdf5 https://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/image.hdf5
```

Verify downloaded files from the repository root:

```powershell
Get-FileHash artifacts/datasets/robomimic/lift/ph/low_dim.hdf5 -Algorithm SHA256
Get-FileHash artifacts/datasets/robomimic/lift/ph/image.hdf5 -Algorithm SHA256
```

The resulting hashes should match the values in the table and the corresponding manifests. Do not add the `.hdf5` files or generated videos to Git.
