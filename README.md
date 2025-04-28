# Scope

Scope is a Object Discovery Model which uses Self Supervised Learning (SSL) Models like DINO to discover unseen objects in images without labeling.

Scope grab the image's attention matrix and represents it a as heatmap image, later it uses Depth models to find the depth of the objects in the image.

Finally, it uses the depth information mixed with the attention matrix to isolate the objects in the image.
