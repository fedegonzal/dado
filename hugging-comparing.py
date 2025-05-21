# DINO v2 Feature Visualization Example (for one image)

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import timm
from torchvision.transforms.functional import to_pil_image
from sklearn.decomposition import PCA
import torch.nn.functional as F
import umap
import seaborn as sns
import numpy as np

# --- Setup ---
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
model.eval()

# Load and preprocess image
def load_image(path):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), img

# 1. Self-Attention Visualization (last layer, head 0)
def visualize_attention(model, image_tensor, layer_idx=-1, head_idx=0):
    attn_map = {}

    def hook_fn(module, input, output):
        # output: [batch, heads, tokens, tokens]
        attn_map['attn'] = output

    # Register hook on attention module of the chosen block
    handle = model.blocks[layer_idx].attn.attn_drop.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)

    handle.remove()  # Clean up

    if 'attn' not in attn_map:
        print("No attention captured")
        return

    attn = attn_map['attn']  # shape: [1, heads, tokens, tokens]
    cls_attn = attn[0, head_idx, 0, 1:]  # CLS token to all patches
    size = int(cls_attn.shape[0] ** 0.5)
    plt.imshow(cls_attn.reshape(size, size).cpu(), cmap='viridis')
    plt.title(f"Attention (Layer {layer_idx}, Head {head_idx})")
    plt.colorbar()
    plt.show()


# 2. Patch Embedding PCA

def visualize_patch_embeddings(model, image_tensor):
    with torch.no_grad():
        feats = model.forward_features(image_tensor)[1:]  # exclude CLS

    pca = PCA(n_components=3)
    proj = pca.fit_transform(feats[0].cpu())
    size = int(proj.shape[0] ** 0.5)
    proj_img = proj.reshape(size, size, 3)
    proj_img = (proj_img - proj_img.min()) / (proj_img.max() - proj_img.min())
    plt.imshow(proj_img)
    plt.title("Patch Embeddings PCA")
    plt.axis("off")
    plt.show()

# 3. Patch Similarity Map

def patch_similarity_map(model, image_tensor, query_index=100):
    with torch.no_grad():
        patches = model.forward_features(image_tensor)[1:]  # exclude CLS

    patch_feats = patches[0]
    query_patch = patch_feats[query_index]
    similarity = F.cosine_similarity(query_patch.unsqueeze(0), patch_feats, dim=1)
    sim_map = similarity.reshape(int(len(similarity) ** 0.5), -1)
    plt.imshow(sim_map.cpu(), cmap='plasma')
    plt.title("Patch Cosine Similarity")
    plt.colorbar()
    plt.show()

# 5. UMAP of Patch Features

def visualize_umap(model, image_tensor):
    with torch.no_grad():
        patches = model.forward_features(image_tensor)[1:][0]

    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(patches.cpu())
    plt.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(len(embedding)), cmap='viridis')
    plt.title("UMAP of Patch Embeddings")
    plt.show()

# Example usage
if __name__ == '__main__':
    image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000030.jpg"
    image_tensor, raw_img = load_image(image_path)

    visualize_attention(model, image_tensor)
    visualize_patch_embeddings(model, image_tensor)
    patch_similarity_map(model, image_tensor, query_index=100)
    visualize_umap(model, image_tensor)
