import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, fpath: str):
        self.fpath = fpath
        self.images = images
        self.poses = poses
        self.focal = focal

    def readFile(self, printShape: bool = False):
        """
        read dataset from a given filepath

        Args:
                printShape: print the shapes of dataset keys

        Returns: tuple -> (images, poses, focal) ; optional shapes
        """
        assert os.path.exists(self.fpath), "check file path"

        dset = np.load(self.fpath)
        self.images = dset["images"]
        self.poses = dset["poses"]
        self.focal = dset["focal"]

        if printShape:
            print(f"Images shape: {images.shape}")
            print(f"Poses shape: {poses.shape}")
            print(f"Focal length: {focal}")

        return images, poses, focal

    def visualize(self, imgIdx: int):
        """
        plot image at a given image index

        Args:
                imgIdx: image index for viewing
        """
        assert imgIdx > 0, "image index must be greater than 0"

        fig, axes = plt.subplots(1, 1, figsize=(17, 10))
        axes.imshow(self.images[imgIdx])
