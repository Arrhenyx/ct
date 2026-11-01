from pathlib import Path

_NIFTI_EXTS = (".nii.gz", ".nii")


def strip_nifti_ext(path: str | Path) -> str:
    """`foo/bar.nii.gz` → `bar`."""
    name = Path(path).name
    for ext in _NIFTI_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem