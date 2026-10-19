import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam2Processor, Sam2Model

ARQUIVO = "nuclear.avif"
MASK_THRESHOLD = 0.0   # negativo (ex. -1.0) = máscara maior, pega fumaça rala
                       # positivo (ex. 1.0)  = só o núcleo denso

image = Image.open(ARQUIVO).convert("RGB")
W, H = image.size
print(f"Imagem: {W}x{H}")

# ---------- 1. Marcar pontos ----------
pos, neg = [], []

def onclick(event):
    if event.inaxes is None or event.xdata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    if event.button == 1:                       # esquerdo = fumaça
        pos.append([x, y])
        plt.plot(x, y, "g*", markersize=14)
    elif event.button == 3:                     # direito = céu / torre
        neg.append([x, y])
        plt.plot(x, y, "rx", markersize=12, mew=3)
    plt.draw()

fig = plt.figure(figsize=(6, 9))
plt.imshow(image)
plt.axis("off")
plt.title("Esq.: fumaça  |  Dir.: céu/torre  |  feche a janela")
fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()

if not pos:  # sem cliques: pontos padrão para esta foto específica
    print("Nenhum clique; usando pontos padrão.")
    pos = [[int(fx * W), int(fy * H)] for fx, fy in
           [(0.345, 0.608), (0.304, 0.162), (0.649, 0.568), (0.568, 0.338)]]
    neg = [[int(fx * W), int(fy * H)] for fx, fy in
           [(0.081, 0.405), (0.527, 0.405), (0.913, 0.270), (0.335, 0.838)]]

points = pos + neg
labels = [1] * len(pos) + [0] * len(neg)
print("points =", points)
print("labels =", labels)

# ---------- 2. Segmentar com SAM 2 ----------
model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large", device_map="auto")
processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

inputs = processor(images=image,
                   input_points=[[points]],     # [imagem][objeto][ponto][x,y]
                   input_labels=[[labels]],     # [imagem][objeto][rótulo]
                   return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)  # 1 máscara (melhor com vários pontos)

masks = processor.post_process_masks(outputs.pred_masks.cpu(),
                                     inputs["original_sizes"],
                                     mask_threshold=MASK_THRESHOLD)[0]
mask = masks[0, 0].numpy().astype(bool)
print(f"Pluma ocupa {100 * mask.mean():.1f}% da imagem")

# ---------- 3. Salvar e mostrar ----------
img = np.array(image)
overlay = img.copy()
overlay[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
Image.fromarray(overlay).save("pluma_mascara.png")
Image.fromarray((mask * 255).astype(np.uint8)).save("pluma_mascara_binaria.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 8))
ax[0].imshow(overlay)
ax[0].set_title("Sobreposição")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Máscara")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()
