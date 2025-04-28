# Loading DINO v2 model (fixed with register tokens)
import torch
from dinov2.models.vision_transformer import vit_small

def load_dino2_model(patch_size, checkpoint, device, img_size=None):
    """ Function printing python version. """

    checkpoint = torch.load(checkpoint, map_location=device)
    del checkpoint['register_tokens'] # we don't need this and will cause an error 

    # Load the model
    model = vit_small(
        patch_size=patch_size,
        img_size=img_size,
        init_values=1.0,
        block_chunks=0
    )


    for p in model.parameters():
        p.requires_grad = False

    state_dict = model.load_state_dict(checkpoint)

    return model
