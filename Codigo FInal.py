import cv2
import numpy as np
from PIL import Image
import os
from maix import nn, camera, display, image
from difflib import SequenceMatcher
import unicodedata
ocr_model_path = "/root/models/pp_ocr_en.mud"
base_path = "/root/teste/"

print("[INFO] Carregando modelo OCR:", ocr_model_path)
ocr = nn.PP_OCR(ocr_model_path)
print("[OK] Modelo OCR carregado com sucesso.")



# ######################################### CV2 INICIO###################################################
# --- Função de debug ---
def save_debug(img, name):
    """Salva a imagem de debug em /root/teste/"""
    if img is None or (hasattr(img, 'size') and img.size == 0):
        print(f"[ERRO] Imagem '{name}' está vazia, não foi salva.")
        return
    path = f"/root/teste/{name}.jpg"
    cv2.imwrite(path, img)
    print(f"Imagem salva em: {path}")

# --- Carregar imagem ---
img_path = "/root/teste/PagT.png"

if not os.path.exists(img_path):
    print(f"[ERRO] Arquivo não encontrado: {img_path}")
    raise SystemExit

ImageTest = cv2.imread(img_path)
if ImageTest is None:
    print(f"[ERRO] Falha ao abrir a imagem: {img_path}")
    raise SystemExit

save_debug(ImageTest, "ImageTest")

# --- Exemplo de detecção de página (substitua pela sua função real) ---
def detect_page(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    save_debug(blur, "Blur")

    edges = cv2.Canny(blur, 75, 200)
    save_debug(edges, "Edges")

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    page_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            page_contour = approx
            break

    if page_contour is None:
        print("[ERRO] Página não detectada.")
        return None, None

    debug_img = cv_img.copy()
    cv2.drawContours(debug_img, [page_contour], -1, (0, 255, 0), 3)
    save_debug(debug_img, "PageContour")

    # Faz uma transformação de perspectiva
    pts = page_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(cv_img, M, (maxWidth, maxHeight))
    save_debug(warped, "WarpedPage")

    return warped, debug_img


#=== remove_border ===
def remove_border(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    graycopy = gray.copy()

    thresh = cv2.threshold(graycopy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    save_debug(morph, "Debug_MorphRemoveborder")

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nenhum contorno encontrado.")
        return img

    big_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(big_contour)
    hullimg = cv2.drawContours(img.copy(), [hull], -1, [0, 255, 45], 3)
    save_debug(hullimg, "Debug_Hull_Removeborder")

    x, y, w, h = cv2.boundingRect(hull)
    imgnoborder = gray[y:y+h, x:x+w]
    save_debug(imgnoborder, "Debug_RemoveBorder_Result")
    return imgnoborder


# === Crop_Image ===
def Crop_Image(page_img):
    blur = cv2.GaussianBlur(page_img, (7, 7), 0)
    save_debug(blur, "Debug_Crop_Blur")

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    save_debug(thresh, "Debug_Crop_Thresh")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    save_debug(dilated, "Debug_Crop_Dilated")

    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours_vis = np.zeros_like(thresh)
    all_contours_vis = cv2.cvtColor(all_contours_vis, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        cv2.drawContours(all_contours_vis, [cnt], -1, (0, 255, 0), 2)
    save_debug(all_contours_vis, "Debug_Crop_AllContours")

    big_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(big_contour)
    hullimg = cv2.drawContours(page_img.copy(), [hull], -1, [0, 255, 45], 3)
    save_debug(hullimg, "Debug_Crop_Hull")

    x, y, w, h = cv2.boundingRect(hull)
    vis_img = page_img[y:y+h, x:x+w]
    save_debug(vis_img, "Debug_Crop_Result")

    return vis_img


# === Crop_Paragraphs ===
def Crop_Paragraphs(vis_img):
    print("\n=== INICIANDO EXTRAÇÃO DE PARÁGRAFOS ===")

    thresh = cv2.adaptiveThreshold(vis_img.copy(), 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    save_debug(thresh, "Debug_Paragraphs_Thresh")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    save_debug(dilated, "Debug_Paragraphs_Dilated")

    _, binary = cv2.threshold(dilated, 50, 255, cv2.THRESH_BINARY_INV)
    save_debug(binary, "Debug_Paragraphs_Binary")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    caixas_longas = []
    LARGURA_MINIMA = 50

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w > LARGURA_MINIMA and area > 100:
            caixas_longas.append((y, y + h))

    caixas_longas.sort()
    recortes = []
    for i in range(len(caixas_longas) - 1):
        _, base_atual = caixas_longas[i]
        _, base_proxima = caixas_longas[i + 1]
        recorte = vis_img.copy()[base_atual:base_proxima, :]
        if recorte.shape[0] > 10:
            recortes.append(recorte)

    print(f"Parágrafos encontrados: {len(recortes)}")

    for i, rec in enumerate(recortes):
        save_debug(rec, f"Debug_Paragraph_{i+1}")

    return recortes





######################################### CV2------------------FINAL ######################################




img_path = "/root/teste/ImageTest.jpg"
ImageTest = cv2.imread(img_path)
if ImageTest is None:
    print("Erro ao ler imagem:", img_path)

save_debug(ImageTest, "ImageTest")
cv_img = ImageTest
noborder = remove_border(cv_img)
vis_img = Crop_Image(noborder)
Crop = Crop_Image(vis_img)
paragrafos = Crop_Paragraphs(Crop)

print("Processamento concluído com sucesso.")
print(f"Total de parágrafos: {len(paragrafos)}")
# --- Detectar página ---
warped_page, debug_img = detect_page(ImageTest)
count = 1






if warped_page is None:
    print("[ERRO] Detecção de página falhou.")
    raise SystemExit

save_debug(debug_img, "Debug_Final")
save_debug(warped_page, "Warped_Final")

print("Processo concluído. Imagens salvas em /root/teste/")









###########################- OCR________________



divisoes = sorted([
    f for f in os.listdir(base_path)
    if f.startswith("Debug_Paragraph_") and f.endswith(".jpg")
])


if not divisoes:
    print("[ERRO] Nenhuma imagem 'divisao_' encontrada para OCR.")
else:
    print(f"[INFO] {len(divisoes)} imagens encontradas para OCR.")

    for nome_img in divisoes:
        caminho_img = os.path.join(base_path, nome_img)
        print(f"[INFO] Processando {caminho_img} ...")

        img = image.load(caminho_img)
        if img is None:
            print(f"[ERRO] Falha ao carregar {nome_img}")
            continue
        print("Formato da imagem:", img.format())
        img = img.to_format(image.Format.FMT_BGR888)
        print("Formato da imagem:", img.format())

        objs = ocr.detect(img)
        objs = list(objs)  # converte o OCR_Objects para lista
        
        objs.sort(key=lambda obj: obj.box.y1)
        texto_final = ""

        for obj in objs:
            texto_final += obj.char_str()
            points = obj.box.to_list()
            img.draw_keypoints(points, image.COLOR_RED, 4, -1, 1)
            img.draw_string(obj.box.x4, obj.box.y4, obj.char_str(), image.COLOR_RED)

        # Salvar imagem com resultado
        img_out_path = os.path.join(base_path, f"OCR_{nome_img}")
        img.save(img_out_path)
        print(f"[OK] Resultado visual salvo em {img_out_path}")

        # Salvar texto
        txt_path = os.path.join(base_path, f"{nome_img}.txt")
        with open(txt_path, "w") as f:
            f.write(texto_final)

        print(f"[OCR] Texto detectado: {texto_final}")
        print(f"[OK] Texto salvo em {txt_path}")


textos_ocr = []
for nome_txt in sorted([
    f for f in os.listdir(base_path)
    if f.startswith("Debug_Paragraph_") and f.endswith(".jpg.txt")
]):
    caminho_txt = os.path.join(base_path, nome_txt)
    with open(caminho_txt, "r") as f:
        textos_ocr.append(f.read().strip())

# --- Junta todos os textos OCR ---
texto_ocr_final = " ".join(textos_ocr)

print("\n--- TEXTO OCR FINAL ---")
print(texto_ocr_final)




print("✅ OCR concluído em todas as divisões.")