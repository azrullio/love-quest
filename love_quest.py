"""
Love Quest - game pixel "tangkap karakter" pakai hand tracking kamera.

Cara main : arahkan tangan ke kamera, lalu CUBIT (jempol + telunjuk) tepat
            di atas karakter untuk menangkapnya.
Stack     : OpenCV (kamera) + MediaPipe Hands (21 titik tangan) + Pygame (game)
Kontrol   : ESC keluar | R restart | D debug on/off | M mode mouse (klik = cubit)
"""
import math
import random

import cv2
import numpy as np
import pygame

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception as exc:  # mediapipe tidak terpasang / versi tanpa solutions API
    mp_hands = None
    print("MediaPipe tidak tersedia, pakai mode mouse:", exc)

# ---------------------------------------------------------------- konfigurasi
WIN_W, WIN_H = 960, 720          # ukuran jendela (4:3, sama dengan webcam)
PIX = 4                          # 1 pixel game = 4x4 pixel layar
LOW_W, LOW_H = WIN_W // PIX, WIN_H // PIX
FPS = 60
MIN_Y, GROUND_Y = 46, 146        # area jalan karakter (koordinat low-res)

LEVELS = [
    # judul, deskripsi, jumlah karakter, kecepatan (px/detik), radius kabur
    ("FIRST HELLO", "Collect the hearts carried by Diran and Dina.", 6, 12, 0),
    ("SECOND DATE", "They walk faster now. Keep up!", 8, 22, 0),
    ("SHY HEARTS", "They run away from your hand. Be quick!", 10, 26, 34),
]

# ---------------------------------------------------------------- pixel art
PALETTE = {
    "H": (196, 74, 104),   # hijab gelap
    "h": (236, 128, 150),  # hijab
    "k": (40, 30, 30),     # rambut
    "s": (244, 204, 170),  # kulit
    "e": (40, 30, 40),     # mata
    "p": (242, 150, 150),  # pipi
    "m": (190, 80, 80),    # mulut
    "r": (214, 52, 52),    # baju merah
    "R": (255, 90, 120),   # hati
    "w": (245, 240, 240),  # celana putih
    "n": (52, 62, 110),    # celana navy
    "b": (60, 40, 40),     # sepatu
}

DINA = [
    "...HHHHHH...",
    "..HhhhhhhH..",
    ".HhssssssHh.",
    ".Hhsesseshh.",
    ".HhpsssspHh.",
    ".HhssmmsshH.",
    "..HhssssHh..",
    ".HhhhhhhhhH.",
    "..rRRrrRRr..",
    ".srRRRRRRrs.",
    "..rrRRRRrr..",
    "..rrrRRrrr..",
    "...wwwwww...",
]
DIRAN = [
    "...kkkkkk...",
    "..kkkkkkkk..",
    "..kksssskk..",
    "..ssessess..",
    "..spsssssp..",
    "...ssmmss...",
    "....ssss....",
    "..rrrrrrrr..",
    "..rRRrrRRr..",
    ".srRRRRRRrs.",
    "..rrRRRRrr..",
    "..rrrRRrrr..",
    "...nnnnnn...",
]
LEGS_A = ["...{0}{0}..{0}{0}...", "...{0}{0}..{0}{0}...", "..bbb..bbb.."]
LEGS_B = ["..{0}{0}....{0}{0}..", "..{0}{0}....{0}{0}..", ".bbb....bbb."]

HEART = [
    ".RR.RR.",
    "RRRRRRR",
    "RRRRRRR",
    ".RRRRR.",
    "..RRR..",
    "...R...",
]


def build_sprite(rows):
    w, h = max(len(r) for r in rows), len(rows)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in PALETTE:
                surf.set_at((x, y), PALETTE[ch])
    return surf


def build_character_frames(body, pants):
    return [build_sprite(body + [row.format(pants) for row in legs])
            for legs in (LEGS_A, LEGS_B)]


# ---------------------------------------------------------------- suara chiptune
def make_sound(freqs, dur=0.06, vol=0.25):
    if not pygame.mixer.get_init():
        return None
    sr, _, channels = pygame.mixer.get_init()
    parts = []
    for f in freqs:
        t = np.linspace(0, dur, int(sr * dur), False)
        wave = np.sign(np.sin(2 * np.pi * f * t)) * vol      # square wave
        parts.append(wave * np.linspace(1, 0.2, len(t)))
    data = (np.concatenate(parts) * 32767).astype(np.int16)
    if channels == 2:
        data = np.column_stack([data, data])
    return pygame.sndarray.make_sound(np.ascontiguousarray(data))


def play(sound):
    if sound:
        sound.play()


# ---------------------------------------------------------------- hand tracking
class HandTracker:
    PINCH_ON, PINCH_OFF = 0.33, 0.48   # hysteresis biar cubitan tidak "kedip"
    SMOOTH = 0.5

    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.hands = None
        if mp_hands:
            self.hands = mp_hands.Hands(max_num_hands=1, model_complexity=0,
                                        min_detection_confidence=0.6,
                                        min_tracking_confidence=0.5)
        self.landmarks = None    # 21 titik (x, y) dalam koordinat layar
        self.cursor = None
        self.pinching = False
        self.pinch_dist = 0.0

    @property
    def ok(self):
        return self.cap.isOpened()

    def read(self):
        """Ambil 1 frame, deteksi tangan, kembalikan frame RGB (sudah di-mirror)."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]                      # crop tengah ke 4:3
        if w / h > 4 / 3:
            nw = int(h * 4 / 3)
            frame = frame[:, (w - nw) // 2:(w - nw) // 2 + nw]
        elif w / h < 4 / 3:
            nh = int(w * 3 / 4)
            frame = frame[(h - nh) // 2:(h - nh) // 2 + nh]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.landmarks = None
        if self.hands:
            rgb.flags.writeable = False
            result = self.hands.process(rgb)
            rgb.flags.writeable = True
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                self.landmarks = [(p.x * WIN_W, p.y * WIN_H) for p in lm]
        self._update_pinch()
        return rgb

    def _update_pinch(self):
        p = self.landmarks
        if not p:
            self.pinching, self.cursor = False, None
            return
        # jarak jempol(4)-telunjuk(8) dibagi ukuran tangan (wrist 0 -> 9),
        # jadi tetap akurat walau tangan dekat/jauh dari kamera
        hand_size = math.dist(p[0], p[9]) or 1.0
        self.pinch_dist = math.dist(p[4], p[8]) / hand_size
        limit = self.PINCH_OFF if self.pinching else self.PINCH_ON
        self.pinching = self.pinch_dist < limit

        tx, ty = (p[4][0] + p[8][0]) / 2, (p[4][1] + p[8][1]) / 2
        if self.cursor is None:
            self.cursor = (tx, ty)
        else:
            cx, cy = self.cursor
            self.cursor = (cx + (tx - cx) * self.SMOOTH, cy + (ty - cy) * self.SMOOTH)

    def release(self):
        self.cap.release()
        if self.hands:
            self.hands.close()


# ---------------------------------------------------------------- objek game
class Character:
    HIT_PAD = 5

    def __init__(self, frames, name, speed, rng):
        self.frames, self.name, self.speed, self.rng = frames, name, speed, rng
        self.x = rng.uniform(14, LOW_W - 14)
        self.y = rng.uniform(MIN_Y + 10, GROUND_Y)
        self.phase = rng.uniform(0, 6.28)
        self.turn_t = 0.0
        self._new_direction()

    def _new_direction(self):
        ang = self.rng.uniform(0, 2 * math.pi)
        self.vx = math.cos(ang) * self.speed
        self.vy = math.sin(ang) * self.speed * 0.6
        self.turn_t = self.rng.uniform(1.5, 3.5)

    def update(self, dt, cursor, flee):
        self.turn_t -= dt
        if self.turn_t <= 0:
            self._new_direction()
        if flee and cursor:
            dx, dy = self.x - cursor[0], self.y - cursor[1]
            d = math.hypot(dx, dy)
            if 0 < d < flee:
                run = self.speed * 2.2
                self.vx, self.vy = dx / d * run, dy / d * run
                self.turn_t = 0.6
        self.x += self.vx * dt
        self.y += self.vy * dt
        if not 10 <= self.x <= LOW_W - 10:
            self.vx = -self.vx
            self.x = min(max(self.x, 10), LOW_W - 10)
        if not MIN_Y <= self.y <= GROUND_Y:
            self.vy = -self.vy
            self.y = min(max(self.y, MIN_Y), GROUND_Y)
        self.phase += dt * (6 + self.speed * 0.25)

    def hit(self, pos):
        w, h = self.frames[0].get_size()
        return (abs(pos[0] - self.x) <= w / 2 + self.HIT_PAD and
                abs(pos[1] - self.y) <= h / 2 + self.HIT_PAD)

    def draw(self, surf):
        frame = self.frames[int(self.phase) % 2]
        bob = int(abs(math.sin(self.phase)) * 2)
        w, h = frame.get_size()
        shadow = pygame.Surface((w, 3), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 70), shadow.get_rect())
        surf.blit(shadow, (self.x - w / 2, self.y + h / 2 - 1))
        surf.blit(frame, (self.x - w / 2, self.y - h / 2 - bob))


class Particle:
    def __init__(self, sprite, x, y, rng):
        self.sprite, self.x, self.y = sprite, x, y
        self.vx, self.vy = rng.uniform(-30, 30), rng.uniform(-60, -25)
        self.life = self.max_life = rng.uniform(0.6, 1.0)

    def update(self, dt):
        self.life -= dt
        self.vy += 50 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        return self.life > 0

    def draw(self, surf):
        img = self.sprite.copy()
        img.set_alpha(int(255 * self.life / self.max_life))
        surf.blit(img, (self.x - 3, self.y - 3))


class Cloud:
    def __init__(self, rng, x=None):
        self.rng = rng
        self.x = rng.uniform(-20, LOW_W + 20) if x is None else x
        self.y = rng.uniform(30, 110)
        self.speed = rng.uniform(2, 5)
        self.blobs = [(rng.randint(-16, 16), rng.randint(-4, 2), rng.randint(5, 10))
                      for _ in range(6)]

    def update(self, dt):
        self.x += self.speed * dt
        if self.x > LOW_W + 30:
            self.__init__(self.rng, x=-30)

    def draw(self, surf):
        for dx, dy, r in self.blobs:
            pygame.draw.circle(surf, (186, 168, 228), (int(self.x + dx), int(self.y + dy + 2)), r)
        for dx, dy, r in self.blobs:
            pygame.draw.circle(surf, (255, 250, 250), (int(self.x + dx), int(self.y + dy)), r)


def build_ground(rng):
    s = pygame.Surface((LOW_W, LOW_H), pygame.SRCALPHA)
    for x in range(LOW_W):
        top = int(GROUND_Y + 10 + 5 * math.sin(x * 0.045) + 3 * math.sin(x * 0.13 + 1))
        pygame.draw.line(s, (96, 120, 40), (x, top), (x, LOW_H))
        pygame.draw.line(s, (150, 170, 66), (x, top), (x, top + 1))
    for _ in range(600):
        x, y = rng.randrange(LOW_W), rng.randrange(GROUND_Y, LOW_H)
        if s.get_at((x, y)).a:
            s.set_at((x, y), rng.choice([(70, 95, 30), (160, 172, 72), (120, 142, 50)]))
    for x in range(LOW_W):
        top = int(LOW_H - 16 + 3 * math.sin(x * 0.08 + 2))
        pygame.draw.line(s, (206, 200, 128), (x, top), (x, LOW_H))
    flowers = [(240, 170, 190), (250, 230, 140), (250, 245, 235), (200, 190, 230)]
    for _ in range(150):
        pygame.draw.circle(s, rng.choice(flowers),
                           (rng.randrange(LOW_W), rng.randrange(LOW_H - 15, LOW_H)),
                           rng.choice([1, 1, 2]))
    return s


# ---------------------------------------------------------------- game utama
class Game:
    CATCH_GRACE = 0.2   # detik setelah mulai cubit, masih boleh menangkap

    def __init__(self):
        self.rng = random.Random()
        self.dina = build_character_frames(DINA, "w")
        self.diran = build_character_frames(DIRAN, "n")
        self.heart = build_sprite(HEART)
        self.ground = build_ground(random.Random(7))
        self.clouds = [Cloud(self.rng) for _ in range(5)]
        self.snd_catch = make_sound([660, 880, 1320])
        self.snd_clear = make_sound([523, 659, 784, 1047], dur=0.1)
        self.restart()

    def restart(self):
        self.level = 0
        self.start_level()

    def start_level(self):
        title, desc, count, speed, flee = LEVELS[self.level]
        self.title, self.desc, self.total, self.flee = title, desc, count, flee
        self.collected = 0
        self.chars = [Character(*((self.diran, "Diran") if i % 2 else (self.dina, "Dina")),
                                speed, self.rng) for i in range(count)]
        self.particles = []
        self.state = "play"
        self.state_t = 0.0
        self.pinch_t = 0.0
        self.was_pinching = False

    def update(self, dt, cursor, pinching):
        for c in self.clouds:
            c.update(dt)
        self.particles = [p for p in self.particles if p.update(dt)]
        self.state_t += dt

        if self.state == "play":
            for c in self.chars:
                c.update(dt, cursor, self.flee)
            if pinching and not self.was_pinching:
                self.pinch_t = 0.0
            self.pinch_t += dt
            if pinching and cursor and self.pinch_t <= self.CATCH_GRACE:
                self._try_catch(cursor)
            if self.collected >= self.total:
                self.state, self.state_t = ("win" if self.level == len(LEVELS) - 1 else "clear"), 0.0
                play(self.snd_clear)
        elif self.state == "clear" and self.state_t > 2.5:
            self.level += 1
            self.start_level()
        self.was_pinching = pinching

    def _try_catch(self, cursor):
        hits = [c for c in self.chars if c.hit(cursor)]
        if not hits:
            return
        target = min(hits, key=lambda c: math.dist((c.x, c.y), cursor))
        self.chars.remove(target)
        self.collected += 1
        self.particles += [Particle(self.heart, target.x, target.y, self.rng) for _ in range(8)]
        self.pinch_t = self.CATCH_GRACE + 1   # 1 cubit = 1 tangkapan
        play(self.snd_catch)

    # ---- gambar layer pixel (low-res, nanti di-scale tanpa blur)
    def draw_pixels(self, surf):
        surf.fill((0, 0, 0, 0))
        for c in self.clouds:
            c.draw(surf)
        surf.blit(self.ground, (0, 0))
        for c in sorted(self.chars, key=lambda c: c.y):
            c.draw(surf)
        for p in self.particles:
            p.draw(surf)
        self._panel(surf, pygame.Rect(4, 4, 132, 32))
        if self.state in ("clear", "win"):
            self._panel(surf, pygame.Rect(LOW_W // 2 - 70, LOW_H // 2 - 22, 140, 40))

    @staticmethod
    def _panel(surf, rect):
        pygame.draw.rect(surf, (200, 70, 110), rect)
        pygame.draw.rect(surf, (255, 208, 222), rect.inflate(-4, -4))
        pygame.draw.rect(surf, (232, 150, 176), rect.inflate(-8, -8), 1)

    # ---- gambar teks (resolusi penuh supaya tajam)
    def draw_text(self, screen, fonts):
        dark, mid = (110, 30, 60), (130, 70, 90)
        blit(screen, fonts["title"], f"LEVEL {self.level + 1}  {self.title}", (34, 30), dark)
        blit(screen, fonts["small"], self.desc, (34, 66), mid)
        blit(screen, fonts["bold"], f"COLLECTED {self.collected}/{self.total}", (34, 98), dark)
        if self.state in ("clear", "win"):
            cx, cy = WIN_W // 2, WIN_H // 2
            big = "LEVEL CLEAR!" if self.state == "clear" else "ALL HEARTS COLLECTED!"
            sub = "Get ready for the next level..." if self.state == "clear" else "Press R to play again"
            blit(screen, fonts["title"], big, (cx, cy - 26), dark, center=True)
            blit(screen, fonts["small"], sub, (cx, cy + 14), mid, center=True)


def blit(screen, font, text, pos, color, center=False, shadow=None):
    if shadow:
        img = font.render(text, True, shadow)
        r = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
        screen.blit(img, r.move(2, 2))
    img = font.render(text, True, color)
    screen.blit(img, img.get_rect(center=pos) if center else img.get_rect(topleft=pos))


def draw_hand(screen, landmarks, cursor, pinching):
    if landmarks and mp_hands:
        for a, b in mp_hands.HAND_CONNECTIONS:
            pygame.draw.line(screen, (90, 220, 110), landmarks[a], landmarks[b], 4)
        for x, y in landmarks:
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 7)
            pygame.draw.circle(screen, (230, 60, 70), (int(x), int(y)), 5)
    if cursor:
        x, y = cursor
        r = 16 if pinching else 12
        pts = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        pygame.draw.polygon(screen, (60, 200, 100) if pinching else (255, 255, 255), pts)
        pygame.draw.polygon(screen, (20, 90, 40), pts, 3)


def main():
    pygame.mixer.pre_init(22050, -16, 1, 512)
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Love Quest - Hand Tracking")
    clock = pygame.time.Clock()
    font = "consolas,couriernew,monospace"
    fonts = {"title": pygame.font.SysFont(font, 24, bold=True),
             "small": pygame.font.SysFont(font, 16),
             "bold": pygame.font.SysFont(font, 17, bold=True)}

    tracker = HandTracker()
    mouse_mode = not tracker.ok or mp_hands is None
    if not tracker.ok:
        print("Kamera tidak terbuka, pakai mode mouse.")
    game = Game()
    layer = pygame.Surface((LOW_W, LOW_H), pygame.SRCALPHA)
    debug = True

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000, 0.05)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                game.restart()
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_d:
                debug = not debug
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_m and tracker.ok and mp_hands:
                mouse_mode = not mouse_mode

        frame = tracker.read() if tracker.ok else None
        if mouse_mode:
            cursor, pinching = pygame.mouse.get_pos(), pygame.mouse.get_pressed()[0]
        else:
            cursor, pinching = tracker.cursor, tracker.pinching
        game.update(dt, (cursor[0] / PIX, cursor[1] / PIX) if cursor else None, pinching)

        # 1) feed kamera
        if frame is not None:
            cam = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
            screen.blit(pygame.transform.smoothscale(cam, (WIN_W, WIN_H)), (0, 0))
        else:
            screen.fill((150, 190, 235))
        # 2) layer pixel art (scale nearest-neighbour = tetap kotak-kotak)
        game.draw_pixels(layer)
        screen.blit(pygame.transform.scale(layer, (WIN_W, WIN_H)), (0, 0))
        # 3) skeleton tangan + kursor + teks
        draw_hand(screen, None if mouse_mode else tracker.landmarks, cursor, pinching)
        game.draw_text(screen, fonts)

        if mouse_mode:
            status = "Mouse mode: click on a character to catch it."
        elif pinching:
            status = "Pinch detected. Grab them!"
        elif tracker.landmarks:
            status = "Hand found. Pinch thumb + index on a character."
        else:
            status = "Show your hand to the camera."
        blit(screen, fonts["bold"], status, (24, WIN_H - 58), (255, 255, 255), shadow=(0, 0, 0))
        if debug and not mouse_mode:
            info = (f"{'PINCH' if pinching else 'OPEN '} | joints: {21 if tracker.landmarks else 0}"
                    f" | pinch distance: {tracker.pinch_dist:.2f} | fps: {clock.get_fps():.0f}")
            blit(screen, fonts["small"], info, (24, WIN_H - 32), (230, 230, 230), shadow=(0, 0, 0))

        pygame.display.flip()

    tracker.release()
    pygame.quit()


if __name__ == "__main__":
    main()
