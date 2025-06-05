import subprocess
import sys


def commit_and_upload_images():
    img_dir = "./data/img"
    try:
        # Track images with git lfs
        subprocess.run(["git", "lfs", "track", f"{img_dir}/*"], check=True)
        # Add .gitattributes if needed
        subprocess.run(["git", "add", ".gitattributes"], check=True)
        # Add images
        subprocess.run(["git", "add", img_dir], check=True)
        # Commit
        subprocess.run(
            ["git", "commit", "-m", "Add/update images in data/img with LFS"],
            check=True,
        )
        # Push (including LFS objects)
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"Error: Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}"
        )
        sys.exit(1)


commit_and_upload_images()
