import math
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter




with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.lower()
alfabet = "aăâbcdefghiîjklmnopqrsștțuvwxyz"
litere = ""
for c in text:
    if c in alfabet:
        litere += c

# a) număr total litere
N = len(litere)
print("nr total litere:", N)

# b) litere distincte
distincte = ""
for c in alfabet:
    if c in litere:
        distincte += c
print("nr distince:", len(distincte))
print("litere dist:", distincte)

# c) număr apariții
numar_aparitii = [0] * len(alfabet)
for lit in litere:
    for i in range(len(alfabet)):
        if lit == alfabet[i]:
            numar_aparitii[i] += 1

print("nr aparitii:")
for i in range(len(alfabet)):
    print(alfabet[i], ":", numar_aparitii[i])

# d) frecvențe
frecvente = [numar_aparitii[i] / N for i in range(len(alfabet))]
print("frecvente:")
for i in range(len(alfabet)):
    print(alfabet[i], ":", round(frecvente[i], 6))

# e) probabilități
p = frecvente[:]

# f) verificare sumă probabilități
suma_p = sum(p)
print("probabilitati:")
for i in range(len(alfabet)):
    print(f"p({alfabet[i]}) = {round(p[i], 6)}")
print("suma p:", round(suma_p, 6))
if abs(suma_p - 1) < 0.0001:
    print("ok")
else:
    print("nu e 1")


# g) Entropia Shannon

H = 0
for prob in p:
    if prob > 0:
        H += prob * math.log2(1 / prob)
print("\nEntropia Shannon H =", round(H, 6), "biți/simbol")


# h) Codare Huffman


# construim coada de priorități (heap)
heap = []
for i in range(len(alfabet)):
    if p[i] > 0:
        heappush(heap, (p[i], alfabet[i]))

# combinăm nodurile cu cele mai mici probabilități
while len(heap) > 1:
    p1, s1 = heappop(heap)
    p2, s2 = heappop(heap)
    heappush(heap, (p1 + p2, [s1, s2]))

# funcție recursivă pt generare coduri Huffman
def genereaza_coduri(tree, prefix="", coduri={}):
    if isinstance(tree, str):
        coduri[tree] = prefix
    else:
        genereaza_coduri(tree[0], prefix + "0", coduri)
        genereaza_coduri(tree[1], prefix + "1", coduri)
    return coduri

# extragem arborele final și generăm codurile
arbore = heappop(heap)[1]
coduri_huffman = genereaza_coduri(arbore)



#  ALGORITM SHANNON–FANO


def shannon_fano(simboluri, probabilitati):
    coduri = {}

    def codare(sf_simboluri, sf_prob, prefix=""):
        if len(sf_simboluri) == 1:
            coduri[sf_simboluri[0]] = prefix
            return

        suma_totala = sum(sf_prob)
        suma = 0
        split_index = 0

        for i, pr in enumerate(sf_prob):
            suma += pr
            if suma >= suma_totala / 2:
                split_index = i + 1
                break

        codare(sf_simboluri[:split_index],
               sf_prob[:split_index],
               prefix + "0")

        codare(sf_simboluri[split_index:],
               sf_prob[split_index:],
               prefix + "1")

    codare(simboluri, probabilitati)
    return coduri


# simboluri și probabilități sortate descrescător
simboluri_sf = []
prob_sf = []

for i in range(len(alfabet)):
    if p[i] > 0:
        simboluri_sf.append(alfabet[i])
        prob_sf.append(p[i])

simboluri_sf, prob_sf = zip(
    *sorted(zip(simboluri_sf, prob_sf), key=lambda x: -x[1])
)

coduri_sf = shannon_fano(list(simboluri_sf), list(prob_sf))


# lungime medie shannon
print("\nCoduri Shannon–Fano:")
lungime_medie_sf = 0

for s in simboluri_sf:
    cod = coduri_sf[s]
    print(s, ":", cod)
    i = alfabet.index(s)
    lungime_medie_sf += len(cod) * p[i]

print("\nLungime medie cod Shannon–Fano =",
      round(lungime_medie_sf, 6), "biți/simbol")

# Eficiență și Redundanță Shannon

eficienta_shannon = H / lungime_medie_sf
redundanta_shannon = 1 - eficienta_shannon

print("Eficiență Shannon =", round(eficienta_shannon * 100, 2), "%")
print("Redundanță Shannon =", round(redundanta_shannon * 100, 2), "%")

# =========================================================
# ARBORE
# =========================================================

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.symbol = None


def build_tree_from_codes(codes):
    root = Node()
    for symbol, code in codes.items():
        node = root
        for bit in code:
            if bit == '0':
                if node.left is None:
                    node.left = Node()
                node = node.left
            else:
                if node.right is None:
                    node.right = Node()
                node = node.right
        node.symbol = symbol
    return root


def compute_positions(node, x=0, y=0, dx=1, pos=None, edges=None):
    if pos is None:
        pos = {}
    if edges is None:
        edges = []

    pos[node] = (x, y)

    if node.left:
        edges.append((node, node.left, '0'))
        compute_positions(node.left, x - dx, y - 1, dx / 2, pos, edges)

    if node.right:
        edges.append((node, node.right, '1'))
        compute_positions(node.right, x + dx, y - 1, dx / 2, pos, edges)

    return pos, edges


def draw_tree(root, title):
    pos, edges = compute_positions(root)

    fig, ax = plt.subplots(figsize=(10, 6))

    for p, c, lbl in edges:
        x1, y1 = pos[p]
        x2, y2 = pos[c]
        ax.plot([x1, x2], [y1, y2])
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, lbl)

    for node, (x, y) in pos.items():
        ax.scatter(x, y)
        label = node.symbol if node.symbol else "*"
        ax.text(x, y + 0.05, label, ha="center")

    ax.set_title(title)
    ax.axis("off")
    ax.invert_yaxis()


# arbore huffman
tree_huffman = build_tree_from_codes(coduri_huffman)
draw_tree(tree_huffman, "Arborele Huffman")


#arbore shannon
tree_sf = build_tree_from_codes(coduri_sf)
draw_tree(tree_sf, "Arborele Shannon–Fano")

# GRAFIC FRECVENȚĂ LITERE

plt.figure(figsize=(12, 6))

# folosim doar literele care apar în text
litere_prezente = []
frecvente_prezente = []

for i in range(len(alfabet)):
    if frecvente[i] > 0:
        litere_prezente.append(alfabet[i])
        frecvente_prezente.append(frecvente[i])

plt.bar(litere_prezente, frecvente_prezente)

plt.xlabel("Litera")
plt.ylabel("Frecvență")
plt.title("Frecvența literelor în text")
plt.grid(axis="y", linestyle="--", alpha=0.6)



print("\nCoduri Huffman:")
lungime_medie = 0
for lit in sorted(coduri_huffman.keys()):
    cod = coduri_huffman[lit]
    print(lit, ":", cod)
    i = alfabet.index(lit)
    lungime_medie += len(cod) * p[i]

print("\nLungime medie cod Huffman =", round(lungime_medie, 6), "biți/simbol")

# Eficiență și Redundanță Huffman

eficienta_huffman = H / lungime_medie
redundanta_huffman = 1 - eficienta_huffman

print("Eficiență Huffman =", round(eficienta_huffman * 100, 2), "%")
print("Redundanță Huffman =", round(redundanta_huffman * 100, 2), "%")



plt.show()