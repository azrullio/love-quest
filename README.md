# Love Quest 💗

Game pixel "tangkap karakter" yang dimainkan pakai **tangan di depan webcam**.
Cubit (jempol + telunjuk) tepat di atas Dina atau Diran untuk mengumpulkan hati yang mereka bawa.

## Tech stack

- **OpenCV** - mengambil frame dari webcam
- **MediaPipe Hands** - mendeteksi 21 titik sendi tangan
- **Pygame** - render game pixel art, suara chiptune, dan HUD

Cubitan dideteksi dari jarak ujung jempol (titik 4) ke ujung telunjuk (titik 8),
dibagi ukuran tangan (titik 0 ke 9), jadi tetap akurat walau tangan dekat/jauh dari kamera.

## Menjalankan

Butuh Python 3.10 / 3.11 (MediaPipe belum mendukung versi yang lebih baru).

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe love_quest.py
```

## Kontrol

| Tombol | Fungsi |
|---|---|
| Cubit di atas karakter | Tangkap |
| `R` | Restart |
| `D` | Debug info on/off |
| `M` | Mode mouse (klik = cubit) |
| `ESC` | Keluar |

## Level

1. **First Hello** - 6 karakter, jalan pelan
2. **Second Date** - 8 karakter, lebih cepat
3. **Shy Hearts** - 10 karakter, kabur kalau tanganmu mendekat

Atur level, kecepatan, dan sensitivitas cubitan (`PINCH_ON`, `SMOOTH`) di bagian atas `love_quest.py`.
