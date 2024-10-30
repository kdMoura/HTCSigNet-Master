from .vit import vit_base_patch16_224_in21k

from .htcsignet import SigTransformer as htcsignet


available_models = {
                    'vit': vit_base_patch16_224_in21k,
                    'htcsignet': htcsignet,
                    }
