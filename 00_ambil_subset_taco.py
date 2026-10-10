"""Membuat dataset deteksi sampah dari TACO (Trash Annotations in Context) dengan kelas pilihan.

Pengganti 00_ambil_subset_coco.py untuk data projek robot trash picking.
Keluarannya SAMA dengan program 00 dosen (dataset_yolo/: classes.txt, images/, labels/),
sehingga program 01, 02, 03, 05, dan 06 bisa langsung dipakai tanpa diubah.

Langkah:
  1. Mengunduh annotations.json TACO (format COCO) dari GitHub.
  2. Memilih gambar yang berisi kelas pilihan (berdasarkan SUPERKATEGORI TACO),
     plus sebagian kecil gambar latar tanpa objek pilihan.
  3. Mengunduh gambar dari Flickr (versi 640 px jika ada), memutar sesuai EXIF,
     dan menyimpannya dengan sisi terpanjang maks. 640 px.
  4. Mengubah bbox COCO (piksel, pojok kiri atas) menjadi label YOLO (cx cy w h, 0-1).

Pemakaian : python 00_ambil_subset_taco.py
            python 00_ambil_subset_taco.py --kelas "Plastic bag & wrapper" Bottle Can --nama plastik botol kaleng
Lanjutan  : python 01_cek_anotasi.py dataset_yolo
Sumber    : https://github.com/pedropro/TACO (lisensi MIT), Proença & Simões, 2020.
"""
import argparse
import io
import json
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

URL_ANOTASI = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
UNDUH = Path("taco_unduhan")
SISI_MAKS = 640

p = argparse.ArgumentParser()
p.add_argument("--kelas", nargs="+", default=["Plastic bag & wrapper", "Bottle", "Can"],
               help="superkategori TACO yang dipakai")
p.add_argument("--nama", nargs="+", default=["plastik", "botol", "kaleng"],
               help="nama kelas baru (urutan sama dengan --kelas)")
p.add_argument("--jumlah", type=int, default=0, help="batasi jumlah gambar berobjek (0 = semua)")
p.add_argument("--latar", type=float, default=0.1, help="proporsi gambar latar tanpa objek (default 0.1)")
p.add_argument("--out", default="dataset_yolo", help="folder keluaran (default dataset_yolo)")
args = p.parse_args()
if len(args.nama) != len(args.kelas):
    raise SystemExit("--nama harus sama banyak dengan --kelas")

UNDUH.mkdir(exist_ok=True)
(UNDUH / "images").mkdir(exist_ok=True)
ann_file = UNDUH / "annotations.json"
if not ann_file.exists():
    print("Mengunduh anotasi TACO ...")
    urllib.request.urlretrieve(URL_ANOTASI, ann_file)
taco = json.loads(ann_file.read_text())

# Superkategori TACO -> class_id YOLO baru
semua_super = sorted({c["supercategory"] for c in taco["categories"]})
salah = [k for k in args.kelas if k not in semua_super]
if salah:
    raise SystemExit(f"Superkategori tidak ada di TACO: {salah}\nPilihan: {', '.join(semua_super)}")
super_ke_baru = {k: i for i, k in enumerate(args.kelas)}
kat_ke_baru = {c["id"]: super_ke_baru[c["supercategory"]]
               for c in taco["categories"] if c["supercategory"] in super_ke_baru}

per_gambar = {}
for a in taco["annotations"]:
    if a["category_id"] in kat_ke_baru and a["bbox"][2] > 0 and a["bbox"][3] > 0:
        per_gambar.setdefault(a["image_id"], []).append(a)

info = {g["id"]: g for g in taco["images"]}
random.seed(0)
berobjek = sorted(per_gambar)
if args.jumlah > 0:
    berobjek = random.sample(berobjek, min(args.jumlah, len(berobjek)))
kandidat_latar = sorted(set(info) - set(per_gambar))
latar = random.sample(kandidat_latar, min(int(len(berobjek) * args.latar), len(kandidat_latar)))
dipilih = berobjek + latar


def nama_file(g):
    return g["file_name"].replace("/", "_")          # batch_1/000006.jpg -> batch_1_000006.jpg


def unduh(img_id):
    """Unduh satu gambar, putar sesuai EXIF, perkecil, simpan. Kembalikan (id, ok, pesan)."""
    g = info[img_id]
    tujuan = UNDUH / "images" / nama_file(g)
    if tujuan.exists():
        return img_id, True, ""
    url = g.get("flickr_640_url") or g["flickr_url"]
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        # Orientasi harus sama dengan ukuran di anotasi (lebar>tinggi atau sebaliknya)
        if (img.width > img.height) != (g["width"] > g["height"]) and img.width != img.height:
            return img_id, False, "orientasi tidak cocok dengan anotasi"
        img.thumbnail((SISI_MAKS, SISI_MAKS))
        img.save(tujuan, quality=92)
        return img_id, True, ""
    except Exception as e:                             # URL mati, timeout, dll.
        return img_id, False, str(e)[:60]


print(f"Kelas dipilih: {dict(zip(args.nama, args.kelas))}")
print(f"Mengunduh {len(dipilih)} gambar dari Flickr (gambar yang sudah ada dilewati) ...")
gagal = []
with ThreadPoolExecutor(16) as ex:
    for n, (img_id, ok, pesan) in enumerate(ex.map(unduh, dipilih), 1):
        if not ok:
            gagal.append((info[img_id]["file_name"], pesan))
        if n % 100 == 0:
            print(f"  {n}/{len(dipilih)}")
gagal_nama = {f for f, _ in gagal}

OUT = Path(args.out)
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True)
(OUT / "labels").mkdir(parents=True)
(OUT / "classes.txt").write_text("\n".join(args.nama) + "\n", encoding="utf-8")

hitung = {k: 0 for k in args.nama}
n_obj = n_latar = 0
for img_id in dipilih:
    g = info[img_id]
    if g["file_name"] in gagal_nama:
        continue
    W, H = g["width"], g["height"]                    # ukuran asli = acuan koordinat bbox
    baris = []
    for a in per_gambar.get(img_id, []):
        x, y, w, h = a["bbox"]
        cid = kat_ke_baru[a["category_id"]]
        cx, cy = min(max((x + w / 2) / W, 0), 1), min(max((y + h / 2) / H, 0), 1)
        baris.append(f"{cid} {cx:.6f} {cy:.6f} {min(w / W, 1):.6f} {min(h / H, 1):.6f}")
        hitung[args.nama[cid]] += 1
    shutil.copy(UNDUH / "images" / nama_file(g), OUT / "images" / nama_file(g))
    stem = Path(nama_file(g)).stem
    (OUT / "labels" / f"{stem}.txt").write_text("\n".join(baris) + ("\n" if baris else ""))
    if baris:
        n_obj += 1
    else:
        n_latar += 1

print(f"\nKelas baru      : {', '.join(f'{i} = {k}' for i, k in enumerate(args.nama))}")
print(f"Gambar berobjek : {n_obj}")
print(f"Gambar latar    : {n_latar}")
print(f"Gagal diunduh   : {len(gagal)}" + (f" (contoh: {gagal[0]})" if gagal else ""))
print("Jumlah kotak    : " + ", ".join(f"{k} {n}" for k, n in hitung.items()))
print(f"Siap di {OUT}/ -> lanjutkan dengan: python 01_cek_anotasi.py {OUT}")
