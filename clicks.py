import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("nuclear.avif").convert("RGB")
print("Tamanho (largura, altura):", img.size)

box, pos, neg = [], [], []

def onclick(event):
    if event.inaxes is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    if len(box) < 2 and event.button == 1:          # 2 primeiros cliques: caixa
        box.append((x, y))
        plt.plot(x, y, "ys", markersize=8)
    elif event.button == 1:                          # esquerdo: fumaça
        pos.append((x, y))
        plt.plot(x, y, "g*", markersize=12)
    elif event.button == 3:                          # direito: fundo
        neg.append((x, y))
        plt.plot(x, y, "rx", markersize=10, mew=3)
    plt.draw()

fig = plt.figure()
plt.imshow(img)
plt.title("2 cliques esq.: cantos da caixa | esq.: fumaça | dir.: fundo | feche ao terminar")
fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()

if len(box) == 2:
    (xa, ya), (xb, yb) = box
    print(f"box = [[[{min(xa,xb)}, {min(ya,yb)}, {max(xa,xb)}, {max(ya,yb)}]]]")
pts = pos + neg
print(f"points = [[{[list(p) for p in pts]}]]")
print(f"labels = [[{[1]*len(pos) + [0]*len(neg)}]]")
