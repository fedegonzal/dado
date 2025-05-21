# Ideas para continuidad de DADO

Dado actual: genera attention, genera depth, separa depth, multiplica attention con cada depth, crea mascaras

## Posibles mejoras

1. Generar depth, separar depth. Generar attention por cada depth, multiplicar, crear mascaras

2. Generar attention, generar patches por entropia. Generar depth, separar depth. Multiplicar, crear mascaras

3. Hacer attention a partir de depth?

3. Evaluar prompts distintos para VLM (llava?)

4. Entrenar un modelo de segmentacion a partir de depth? usando dino2 como backbone

5. Entrenar un modelo de deteccion de objetos para una sola clase "OBJETO" (backbone) y luego sumar lo nuestro 

6. Watershed segmentation para segmentar objetos unidos

PCA
Attention
Depth
Entropy

Superpixel -> slic + quickshift + felzenszwalb + quickshift
active_contour + inverse_gaussian_gradient + morphological_chan_vese
chan_vese + morphological_chan_vese
flood
random_walker
Watershed segmentation

7. Region adjacency graph (RAG) Thresholding para unificar el depth

8. https://chatgpt.com/c/682aea10-0178-8007-8ad2-758ea74d8fcc


Evaluar modelos depth: calidad, detalle, velocidad, etc.

mask slic
https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_mask_slic.html#sphx-glr-auto-examples-segmentation-plot-mask-slic-py

Region Adjacency Graph (RAG)
https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_ncut.html#sphx-glr-auto-examples-segmentation-plot-ncut-py
https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_rag_mean_color.html#sphx-glr-auto-examples-segmentation-plot-rag-mean-color-py

