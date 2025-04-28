
import torch
from dinov1.vision_transformer import vit_small

def load_dino1_model(patch_size, checkpoint, device, img_size=None):
    """ Function printing python version. """

    checkpoint = torch.load(checkpoint, map_location=device)

    # Load the model
    model = vit_small(
        patch_size=patch_size,
        init_values=1.0,
        block_chunks=0
    )


    for p in model.parameters():
        p.requires_grad = False

    state_dict = model.load_state_dict(checkpoint)

    return model
