import sys

#sys.path.append("/home/htcsignet/")
import argparse
import functools
from typing import Tuple, Optional
import numpy as np

from htcsignet.datasets import available_datasets
from htcsignet.datasets.base import IterableDataset
from htcsignet.datasets.util import process_dataset_images

from htcsignet.preprocessing.normalize import preprocess_signature


def process_dataset(dataset: IterableDataset,
                    save_path: str,
                    img_size: Tuple[int, int],
                    subset: Optional[slice] = None):
    """ Processes a dataset (normalizing the images) and saves the result as a
        numpy npz file (collection of np.ndarrays).

    Parameters
    ----------
    dataset : IterableDataset
        The dataset, that knows where the signature files are located
    save_path : str
        The name of the file to save the numpy arrays
    img_size : tuple (H x W)
        The final size of the images
    subset : slice
        Which users to consider. e.g. slice(None) to consider all users, or slice(first, last)


    Returns
    -------
    None

    """
    preprocess_fn = functools.partial(preprocess_signature,
                                      canvas_size=dataset.maxsize,
                                      img_size=img_size,
                                      input_size=img_size)  # Don't crop it now

    if subset is None:
        subset = slice(None)  # Use all
    processed = process_dataset_images(dataset, preprocess_fn, img_size, subset)
    x, y, yforg, user_mapping, used_files = processed

    np.savez(save_path,
             x=x,
             y=y,
             yforg=yforg,
             user_mapping=user_mapping,
             filenames=used_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('--dataset', choices=available_datasets.keys(), default='gpds',
                        help='The dataset type')
    parser.add_argument('--path', default='../../data/GPDS/GPDS1_1000',
                        help='Path to the folder containing the signatures')
    parser.add_argument('--save-path', default="GPDS_1000_256X256.npz",
                        help='Path to save the numpy arrays')
    parser.add_argument('--image-size', nargs=2, type=int, default=(256, 256),
                        help='Image size (H x W)')

    args = parser.parse_args()

    ds = available_datasets[args.dataset]
    dataset = ds(args.path)

    print('Processing dataset')
    process_dataset(dataset, args.save_path, args.image_size)
