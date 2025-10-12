import re
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

"""
Stitch 2D segmentation slices back into 3D volumes.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", required=True)
    p.add_argument("--dest_folder", required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--grp_regex", required=True)
    p.add_argument("--source_scan_pattern", required=True)
    p.add_argument("--recursive", action="store_true")
    return p.parse_args()


def load_png(path):
    #  Helper method to load png images
    image = Image.open(path)
    array = np.array(image)
    if array.ndim == 3:
        return array[..., 0]
    else:
        return array


def to_class_ids(a, k):
    #  Helper method to convert to the required number of classes
    if k <= 255 and a.max() > (k - 1):
        a = np.rint(a / (255.0 / (k - 1)))
    return a.astype(np.int64)


def resize_label(a, X, Y):
    #  Helper method to resize label images to ensure they fit the target volume
    return np.array(
        Image.fromarray(a.astype(np.uint16 if a.dtype.itemsize > 1 else np.uint8)).resize((X, Y), Image.NEAREST))


def stitch(slices, idxs, shape, dtype):
    # Stitch the 2D slices into a 3D volume
    X, Y, Z = shape
    volume = np.zeros((X, Y, Z), dtype=dtype)
    for slice, i in zip(slices, idxs):
        slice = np.squeeze(slice)
        if slice.shape != (X, Y):
            slice = resize_label(slice, X, Y)
        volume[:, :, i] = slice.astype(dtype, copy=False)
    return volume


def main():
    args = parse_args()
    data_dir = Path(args.data_folder)
    dest_dir = Path(args.dest_folder);
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_dtype = np.uint8 if args.num_classes <= 255 else np.uint16
    group_pattern = re.compile(args.grp_regex)
    inedx_pattern = re.compile(r"_(\d{4})(?!\d)")

    # Get needed files
    file_search = data_dir.rglob if args.recursive else data_dir.glob
    files = sorted(file_search("*.png")) + sorted(file_search("*.nii.gz"))

    groups = {}
    for file in files:
        patient_id = group_pattern.search(file.name).group(1)
        groups.setdefault(patient_id, []).append(file)
    # Loop over files and stitch them
    for patient_id, file_sort in groups.items():
        file_sort.sort(key=lambda p: int(inedx_pattern.search(p.stem).group(1)))
        ref_path = args.source_scan_pattern.format(id_=patient_id)
        ref_img = nib.load(ref_path)
        X, Y, Z = np.asanyarray(ref_img.dataobj).shape

        slices, indexes = [], []
        for file in file_sort:
            array = load_png(str(file)) if file.suffix.lower() == ".png" else np.squeeze(np.asanyarray(nib.load(str(file)).dataobj))
            array = np.clip(to_class_ids(array, args.num_classes), 0, args.num_classes - 1)
            slices.append(array)
            indexes.append(int(inedx_pattern.search(file.stem).group(1)))

        header = ref_img.header.copy()
        header.set_data_dtype(out_dtype)
        volume = stitch(slices, indexes, (X, Y, Z), out_dtype)
        nib.save(nib.Nifti1Image(volume, ref_img.affine, header=header), str(dest_dir / f"{patient_id}.nii.gz"))


if __name__ == "__main__":
    main()
