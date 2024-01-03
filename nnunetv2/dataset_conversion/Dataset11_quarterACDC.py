import os
import shutil
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def create_path(path):
    if os.path.exists(path) is True:
        shutil.rmtree(path)
    os.mkdir(path)
    return path


def make_out_dirs(dataset_id: int, task_name="quarter_ACDC"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    create_path(out_dir)
    create_path(out_train_dir)
    create_path(out_labels_dir)
    create_path(out_test_dir)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):

    patients_train = sorted([f for f in (src_data_folder / "training").iterdir()])
    patients_label = sorted([f for f in (src_data_folder / "label").iterdir()])
    patients_test = sorted([f for f in (src_data_folder / "testing").iterdir()])

    num_training_cases = 0
    num_label_cases = 0
    num_test_cases = 0

    for nii_file in patients_train:
        shutil.copy(nii_file, train_dir / f"{num_training_cases:04d}_0000.nii.gz")
        num_training_cases += 1

    for nii_file in patients_label:
        shutil.copy(nii_file, labels_dir / f"{num_label_cases:04d}.nii.gz")
        num_label_cases += 1

    for nii_file in patients_test:
        shutil.copy(nii_file, test_dir / f"{num_test_cases:04d}_0000.nii.gz")
        num_test_cases += 1

    if num_training_cases != num_label_cases:
        raise ValueError("num_training_cases != num_label_cases")

    return num_training_cases


def convert_acdc(src_data_folder: str, dataset_id):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={
            "background": 0,
            "target": 1,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


if __name__ == "__main__":
    quarter_acdc_src = r"C:\Git\DataSet\abdomen\temp"

    convert_acdc(quarter_acdc_src, 11)
