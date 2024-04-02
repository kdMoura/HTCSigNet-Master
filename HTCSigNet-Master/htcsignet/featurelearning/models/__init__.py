from .vit import vit_base_patch16_224_in21k

from .cnn_vit_sig import SigTransformer


available_models = {
                    'vit': vit_base_patch16_224_in21k,
                    'cnn_vit': Sig_Transformer,
                    }
