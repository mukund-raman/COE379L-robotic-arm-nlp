import os
import getpass
import subprocess
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F
import argparse
REMOTE_MACHINE = True  # Set to True if running on a remote machine

def load_and_parse_coco_dataset(classes, max_samples=500, \
    export_dir="../data/coco_data"):
    """
    Loads the COCO-2017 dataset, filters it by the specified classes,
    and exports it.

    Args:
        classes (list): List of class names to filter.
        max_samples (int): Maximum number of samples to load.
        export_dir (str): Directory to export the filtered dataset.
    """
    # Load the COCO-2017 dataset
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        download_if_necessary=True,
        shuffle=True,
    )

    # Filter and export the dataset and compute uniqueness of samples
    filtered_view = dataset.filter_labels("ground_truth", F("label").is_in(classes))
    filtered_view = filtered_view.match(F("ground_truth.detections").length() > 0)
    os.makedirs(export_dir, exist_ok=True)
    filtered_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
    )
    fob.compute_uniqueness(filtered_view)
    
    # Write all the classes to a classes.txt file in the export directory
    with open(os.path.join(export_dir, "classes.txt"), "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
            
    # Move files in images/train to just images and remove images/train,
    # doing the same for labels/train in the export directory
    os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "labels"), exist_ok=True)
    for file in os.listdir(os.path.join(export_dir, "images/train")):
        os.rename(os.path.join(export_dir, "images/train", file), \
            os.path.join(export_dir, "images", file))
    for file in os.listdir(os.path.join(export_dir, "labels/train")):
        os.rename(os.path.join(export_dir, "labels/train", file), \
            os.path.join(export_dir, "labels", file))
    os.rmdir(os.path.join(export_dir, "images/train"))
    os.rmdir(os.path.join(export_dir, "labels/train"))

    # Print dataset summary and example
    print(f"Loaded {len(filtered_view)} samples from the COCO-2017 dataset \
        with the following classes: {classes}:")
    print(filtered_view)
    print("\nHere is the first sample in the dataset:")
    print(filtered_view.first())

    return filtered_view

def setup_ngrok():
    """
    Ensures ngrok is installed and available for use.
    """
    # Install ngrok if not already installed
    if not os.path.exists("/usr/local/bin/ngrok"):
        print("Downloading and setting up ngrok...")
        try:
            # Download ngrok
            password = getpass.getpass("Enter your sudo password: ")
            script = """
            echo "{password}" | sudo -S bash -c '
            curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
            | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
            && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
            | tee /etc/apt/sources.list.d/ngrok.list \
            && apt update \
            && apt install -y ngrok'
            """.format(password=password)
            subprocess.run(script, shell=True, check=True, executable="/bin/bash")
        except subprocess.CalledProcessError as e:
            print(f"Error during ngrok setup: {e}")
            raise
    else:
        print("ngrok is already installed. Skipping setup.")
    
    # Add ngrok authentication token
    try:
        subprocess.run(["ngrok", "config", "add-authtoken", \
            "2wetcAzgCSHHbq7vJ59nsff61MD_4uPVdNp1G7SJQfhGT1ePA"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error adding ngrok auth token: {e}")
        raise
    else:
        print("ngrok is already authenticated.")

def launch_fiftyone_app(filtered_view):
    """
    Launches the FiftyOne app to view the dataset, with optional remote
    access via ngrok.

    Args:
        filtered_view: The filtered FiftyOne dataset view.
    """
    try:
        setup_ngrok()
        if REMOTE_MACHINE:
            print("Starting ngrok on port 5151...")
            ngrok_process = subprocess.Popen(["ngrok", "http", \
                "http://localhost:5151"])
        session = fo.launch_app(filtered_view, port=5151)
        print(f"FiftyOne running at: {session.url}")
    except subprocess.CalledProcessError as e:
        print(f"Error starting ngrok: {e}")
        raise
    except Exception as e:
        print(f"Error launching FiftyOne app: {e}")
        raise
    finally:
        if REMOTE_MACHINE:
            ngrok_process.terminate()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load and parse the \
        COCO-2017 dataset.")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="List of class names to filter (e.g., --classes bottle cat dog).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to load (default: 500).",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="./coco_data",
        help="Directory to export the filtered dataset (default: \
            ../data/coco_data).",
    )
    args = parser.parse_args()

    # Load and parse the COCO dataset
    filtered_view = load_and_parse_coco_dataset(
        classes=args.classes,
        max_samples=args.max_samples,
        export_dir=args.export_dir
    )

    # Launch the FiftyOne app
    launch_fiftyone_app(filtered_view)