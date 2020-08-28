import os
import pytest
import subprocess
import yaml

from semeio.jobs import ensure_folder_structure


def _create_folders(folders):
    for folder in folders:
        os.makedirs(folder)


def assert_leaf_folders(folders, base="."):
    base = os.path.realpath(base)
    leaf_folders_on_disk = []
    for root, dirs, _ in os.walk(base):
        if len(dirs) == 0 and root != base:
            leaf_folders_on_disk.append(root)

    leaf_folders_on_disk = sorted(map(os.path.realpath, leaf_folders_on_disk))
    folders = sorted([os.path.realpath(os.path.join(base, elem)) for elem in folders])

    assert leaf_folders_on_disk == folders


@pytest.mark.parametrize(
    ("config", "existing_folders", "expected_new_folders", "root"),
    [
        ["a:", [], ["a"], ".",],
        [
            """
              a:
                b:
                c:
            """,
            ["a/b"],
            ["a/c"],
            ".",
        ],
        [
            """
              a:
                b:
                c:
            """,
            [],
            ["root/a/b", "root/a/c"],
            "root",
        ],
        [
            """
              a:
                b:
                c:
            """,
            ["root/a/b"],
            ["root/a/c"],
            "root",
        ],
        [
            """
            in:
              some_data:
              the_secret_sauce:
                ingredient1:
                ingredient2:
            config:
            out:
              exports:
              storage:
            """,
            [],
            [
                "in/some_data",
                "in/the_secret_sauce/ingredient1",
                "in/the_secret_sauce/ingredient2",
                "config",
                "out/exports",
                "out/storage",
            ],
            ".",
        ],
    ],
)
def test_ensure_folder_structure(
    config, existing_folders, expected_new_folders, root, tmpdir,
):
    tmpdir.chdir()
    config = yaml.safe_load(config)

    _create_folders(existing_folders)
    assert_leaf_folders(existing_folders)

    ensure_folder_structure.run(config, root)
    assert_leaf_folders(existing_folders + expected_new_folders)


def test_ensure_folder_structure_backroot(tmpdir):
    subdir = tmpdir.mkdir("sub")
    subdir.chdir()

    config = {"a": {"b": None, "c": None}}
    root = ".."

    ensure_folder_structure.run(config, root)
    assert_leaf_folders(["a/b", "a/c", "sub"], base="..")


def test_ensure_folder_structure_existing_files(tmpdir):
    tmpdir.chdir()
    tmpdir.mkdir("a")

    with open("a/text", "w") as f:
        f.write("Hello")

    config = {"a": {"b": None, "c": None}}
    ensure_folder_structure.run(config, ".")

    with open("a/text") as f:
        assert f.readline() == "Hello"


def test_ensure_folder_structure_file_collision(tmpdir):
    config = {"a": {"text": None, "c": None}}

    with open("a/text", "w") as f:
        f.write("Hello")

    with pytest.raises(FileExistsError):
        ensure_folder_structure.run(config, ".")

    with open("a/text") as f:
        assert f.readline() == "Hello"


def test_ensure_folder_structure_abspath(tmpdir):
    tmpdir.chdir()
    config = {"a": {"/b": None, "c": None}}
    with pytest.raises(ValueError) as exp:
        ensure_folder_structure.run(config, ".")

    assert "Configured structure cannot contain absolute elements" in str(exp.value)


def test_ensure_folder_structure_back(tmpdir):
    tmpdir.chdir()
    config = {"a": {"..": None, "c": None}}
    with pytest.raises(ValueError) as exp:
        ensure_folder_structure.run(config, ".")

    assert "Path elements should not contain '..'" in str(exp.value)


def test_ensure_folder_structure_script(tmpdir):
    tmpdir.chdir()

    with open("config.yml", "w") as f:
        f.write(
            """
            a:
              b:
              c:
            d:
        """
        )

    subprocess.check_call(["ensure_folder_structure.py", "config.yml"])

    assert_leaf_folders(["a/b", "a/c", "d"], ".")
