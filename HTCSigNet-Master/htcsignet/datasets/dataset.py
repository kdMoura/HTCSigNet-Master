import os
from skimage.io import imread
from sigver.datasets.base import IterableDataset
from skimage import img_as_ubyte


class GPDSDataset(IterableDataset):
    """ Helper class to load the GPDS-960 Grayscale dataset
    """

    def __init__(self, path, extension='png'):
        self.path = path
        self.users = [int(user) for user in sorted(os.listdir(self.path))]
        self.extension = extension

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 30

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 952, 1360

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        user_folder = os.path.join(self.path, '{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_genuine_files = filter(lambda x: x[0:2] + x[-3:] == 'c-jpg', all_files)
        for f in user_genuine_files:
            full_path = os.path.join(user_folder, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        user_folder = os.path.join(self.path, '{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_forgery_files = filter(lambda x: x[0:2] + x[-3:] == 'cfjpg', all_files)
        for f in user_forgery_files:
            full_path = os.path.join(user_folder, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f


    def get_signature(self, user, img_idx, forgery):
        """ Returns a particular signature (given by user id, img id and
            whether or not it is a forgery
        """

        if forgery:
            prefix = 'cf'
        else:
            prefix = 'c'
        filename = '{}-{:03d}-{:02d}.{}'.format(prefix, user, img_idx,
                                                self.extension)
        full_path = os.path.join(self.path, '{:03d}'.format(user), filename)
        return img_as_ubyte(imread(full_path, as_gray=True))

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries


# 以下几个数据集的处理需要你参照上面的修改一下
class CedarDataset(IterableDataset):
    """ Helper class to load the CEDAR dataset
    """

    def __init__(self, path):
        self.path = path #"E:\研究生\学习资料\签名鉴定\几个数据集\CEDAR"
        self.users = list(range(1, 55 + 1))

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 24

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 952, 1360

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}_{}_{}.png'.format('original', user, img) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'full_org', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}_{}_{}.png'.format('forgeries', user, img) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'full_forg', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries


class UTSigDataset(IterableDataset):
    """ Helper class to load the UTSig dataset
    """

    def __init__(self):
        self.path = "E:\研究生\学习资料\签名鉴定\几个数据集/UTSig"
        self.users = list(range(1, 115 + 1))

    @property
    def genuine_per_user(self):
        return 27

    @property
    def skilled_per_user(self):
        return 6

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 1500, 2100

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}/{}.tif'.format(user, img) for img in range(1, 27 + 1)]
        for f in files:
            full_path = os.path.join(self.path, "Genuine", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}/{}.tif'.format(user, img) for img in range(1, 6 + 1)]
        for f in files:
            full_path = os.path.join(self.path, "Forgery", "Skilled", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries


class BHSigBDataset(IterableDataset):
    """ Helper class to load the UTSig dataset
    """

    def __init__(self, path):
        self.path = path
        self.users = list(range(1, 100 + 1))

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 30

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 310, 1100

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}{}{}/B-S-{}-G-{}{}.tif'.format(int(user / 100 % 10), int(user / 10 % 10), int(user % 10), user,
                                                   int(img / 10 % 10), int(img % 10)) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path, "Bengali", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}{}{}/B-S-{}-F-{}{}.tif'.format(int(user / 100 % 10), int(user / 10 % 10), int(user % 10), user,
                                                   int(img / 10 % 10), int(img % 10)) for img in range(1, 30 + 1)]
        for f in files:
            full_path = os.path.join(self.path, "Bengali", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries


class BHSigHDataset(IterableDataset):
    """ Helper class to load the UTSig dataset
    """

    def __init__(self, path):
        self.path = path
        self.users = list(range(1, 160 + 1))

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 30

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 450, 1350

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""
        
        user_folder = os.path.join(self.path, 'Hindi','{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_genuine_files = filter(lambda x: '-G-' in x, all_files)
        for f in user_genuine_files:
            full_path = os.path.join(user_folder, f)
            #print('DEBUG: ',full_path)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        user_folder = os.path.join(self.path, 'Hindi','{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_forgery_files = filter(lambda x: '-F-' in x, all_files)
        for f in user_forgery_files:
            full_path = os.path.join(user_folder, f)
            #print('DEBUG: ',full_path)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
