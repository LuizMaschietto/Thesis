from maix import camera, display, gpio, pinmap, image, time
import os
import cv2
import numpy as np

# === Inicializações ===
print("Inicializando câmera em 1920x1080...")
cam = camera.Camera(1920, 1080)
disp = display.Display()

# === Configuração do botão no pino A19 (GPIOA19) com pull-up ===
pinmap.set_pin_function("A19", "GPIOA19")
botao = gpio.GPIO("GPIOA19", gpio.Mode.IN, gpio.Pull.PULL_UP)

# === Loop de preview aguardando botão ===
print("Mostrando preview. Pressione o botão (A19) para capturar.")
while botao.value() == 1:  # botão solto
    img = cam.read()
    disp.show(img)
    time.sleep_ms(20)

# === Captura a imagem após o botão ser pressionado ===
img = cam.read()
img_path = "/sd/pagina.jpg"
img.save(img_path)
print("Imagem capturada e salva em:", img_path)

# === Processamento com OpenCV ===
cv_img = cv2.imread(img_path)
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

if len(contours) == 0:
    print("Nenhuma página detectada.")
    raise SystemExit

# Assume que o maior contorno é a página
x, y, w, h = cv2.boundingRect(contours[0])
page_img = cv_img[y:y+h, x:x+w]
cv2.rectangle(cv_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite("/sd/debug_page_detected.jpg", cv_img)

# === Pré-processamento para encontrar estrofes ===
gray_page = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
thresh_page = cv2.adaptiveThreshold(gray_page, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
morph = cv2.dilate(thresh_page, kernel, iterations=1)

blocks, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blocks_sorted = sorted(blocks, key=lambda c: cv2.boundingRect(c)[1])  # ordena verticalmente

# Cria diretório para salvar estrofes
estrofe_dir = "/sd/estrofes"
if not os.path.exists(estrofe_dir):
    os.mkdir(estrofe_dir)

vis_img = page_img.copy()

for i, cnt in enumerate(blocks_sorted):
    bx, by, bw, bh = cv2.boundingRect(cnt)
    estrofe = page_img[by:by+bh, bx:bx+bw]
    save_path = f"{estrofe_dir}/estrofe_{i}.jpg"
    cv2.imwrite(save_path, estrofe)
    print(f"Estrofe {i} salva em: {save_path}")
    cv2.rectangle(vis_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
    cv2.putText(vis_img, f"{i}", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("/sd/debug_estrofes_detectadas.jpg", vis_img)

print("Processamento finalizado com sucesso.")
