from .dataset import *

available_datasets = {'gpds': GPDSDataset,
                      'cedar':CedarDataset,
                        'utsig':UTSigDataset,
                        'bhsig_b':BHSigBDataset,
                        'bhsig_h':BHSigHDataset}
