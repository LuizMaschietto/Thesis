from maix import camera, image, display
import cv2
import os
import numpy as np

# Inicializa câmera e display
cam = camera.Camera(1920, 1080)
disp = display.Display()

# Captura uma imagem da câmera
img = cam.read()

# Salva a imagem capturada no SD
img_path = "/sd/pagina.jpg"
img.save(img_path)
print("Imagem capturada e salva em:", img_path)

# Carrega a imagem usando OpenCV
cv_img = cv2.imread(img_path)

# Converte para escala de cinza e aplica blur + limiarização adaptativa
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

# Detecta contornos externos (para encontrar a página)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Assume que o maior contorno é a página
x, y, w, h = cv2.boundingRect(contours[0])
page_img = cv_img[y:y+h, x:x+w]
cv2.rectangle(cv_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Salva visualização da página detectada
cv2.imwrite("/sd/debug_page_detected.jpg", cv_img)

# Converte a página para cinza e aplica threshold
gray_page = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
thresh_page = cv2.adaptiveThreshold(gray_page, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

# linhas em blocos (estrofes)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
morph = cv2.dilate(thresh_page, kernel, iterations=1)

# Encontra contornos dos blocos de texto
blocks, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blocks_sorted = sorted(blocks, key=lambda c: cv2.boundingRect(c)[1])  # ordena por posição vertical

# Cria diretório de saída, se necessário
estrofe_dir = "/sd/estrofes"
if not os.path.exists(estrofe_dir):
    os.mkdir(estrofe_dir)

# Cópia para visualização dos blocos
vis_img = page_img.copy()

# Salva cada estrofe como imagem separada
for i, cnt in enumerate(blocks_sorted):
    bx, by, bw, bh = cv2.boundingRect(cnt)
    estrofe = page_img[by:by+bh, bx:bx+bw]
    save_path = f"{estrofe_dir}/estrofe_{i}.jpg"
    cv2.imwrite(save_path, estrofe)
    print(f"Estrofe {i} salva em: {save_path}")

    # Desenha retângulo e índice
    cv2.rectangle(vis_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
    cv2.putText(vis_img, f"{i}", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Salva visualização final com estrofes detectadas
cv2.imwrite("/sd/debug_estrofes_detectadas.jpg", vis_img)

print("Processamento finalizado com sucesso.")
