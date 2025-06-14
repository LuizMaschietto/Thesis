from maix import camera, display, gpio, pinmap, image, time
import os
import cv2
import numpy as np

# === Initialize camera and display ===
print("Initializing camera at 1920x1080...")
cam = camera.Camera(720, 600)
disp = display.Display()

# === Setup button pin A19 (GPIOA19) with pull-up ===
pinmap.set_pin_function("A19", "GPIOA19")
button = gpio.GPIO("GPIOA19", gpio.Mode.IN, gpio.Pull.PULL_UP)

def wait_for_button_press():
    print("Showing preview. Press the button (A19) to capture.")
    while button.value() == 1:  # button not pressed
        img = cam.read()
        disp.show(img)
        time.sleep_ms(20)
    # Debounce delay
    time.sleep_ms(200)
    while button.value() == 0:  # wait for button release
        time.sleep_ms(20)

def order_points(pts):
    # Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    # Compute height of new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for bird-eye view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_page(cv_img):
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Blur to smooth noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold + invert to get white text on black
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found!")
        return None, None

    # Find contour with largest area that approximates a quadrilateral (page-like)
    page_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:  # skip small contours
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            page_contour = approx

    if page_contour is None:
        print("No page-like contour found!")
        return None, None

    # Draw detected page contour on debug image
    debug_img = cv_img.copy()
    cv2.drawContours(debug_img, [page_contour], -1, (0, 255, 0), 4)
    cv2.imwrite("/sd/debug_page_contour.jpg", debug_img)
    print("Page contour detected and saved in debug_page_contour.jpg")

    # Perspective transform to get top-down view of page
    warped = four_point_transform(cv_img, page_contour.reshape(4, 2))

    return warped, debug_img


def detect_paragraphs(page_img):
    # Convert to grayscale
    gray_page = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray_page, (7, 7), 0)

    # Thresholding with Otsu (no adaptive here to keep things simpler)
    _, thresh_page = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert threshold because text is usually dark on light background
    thresh_page = cv2.bitwise_not(thresh_page)

    cv2.imwrite("/sd/debug_Thresh.jpg", thresh_page)

    # Find contours on threshold image
    contours, _ = cv2.findContours(thresh_page.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = thresh_page.shape

    # Find largest contour that touches image border (likely the black border)
    border_contour = None
    max_area = 0

    border_threshold = 5  # pixels from the edge
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        touches_edge = (
            x <= border_threshold or 
            y <= border_threshold or 
            (x + w) >= (w_img - border_threshold) or 
            (y + h) >= (h_img - border_threshold)
        )

        if touches_edge and area > max_area:
            max_area = area
            border_contour = cnt

    # Create mask to remove just the border contour
    mask = np.ones(thresh_page.shape, dtype=np.uint8) * 255  # white mask initially

    if border_contour is not None:
        # Fill in the border contour area with 0 (black) on mask to remove it
        cv2.drawContours(mask, [border_contour], -1, 0, thickness=cv2.FILLED)

    # Use mask to remove border contour in threshold image
    thresh_clean = cv2.bitwise_and(thresh_page, thresh_page, mask=mask)

    cv2.imwrite("/sd/debug_Thresh_no_border.jpg", thresh_clean)

    # Morphological dilation on cleaned threshold image to connect paragraph lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_clean = cv2.dilate(thresh_clean, kernel, iterations=4)
    cv2.imwrite("/sd/debug_Dilated.jpg", dilated_clean)

    # Find contours on dilated cleaned image (paragraph blocks)
    contours, _ = cv2.findContours(dilated_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Detected {len(contours)} paragraph contours after border removal")

    if len(contours) == 0:
        print("No paragraphs detected!")
        return [], page_img.copy(), dilated_clean

    # Sort contours top-to-bottom
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    vis_img = page_img.copy()
    paragraphs = []
    for i, cnt in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio < 5 and area > 120:  # Adjust thresholds as needed
            paragraph = page_img[y:y+h, x:x+w]
            save_path = f"/sd/paragraph_{i}.jpg"
            cv2.imwrite(save_path, paragraph)
            paragraphs.append(save_path)
            print(f"Paragraph {i} saved in: {save_path}")
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite("/sd/debug_paragraphs_detected.jpg", vis_img)
    print("Paragraph detection completed and saved in debug_paragraphs_detected.jpg")

    return paragraphs, vis_img, dilated_clean



# Main execution
wait_for_button_press()

# Capture image
img = cam.read()
if img is None:
    print("Error capturing image.")
    raise SystemExit
img_path = "/sd/captured_page.jpg"
img.save(img_path)
img_Test="/sd/ImagemTParagrafo.png"
print("Image captured and saved in:", img_path)

# Process the captured image
cv_img = cv2.imread(img_Test)
if cv_img is None:
    print("Error reading image with OpenCV.")
    raise SystemExit

# Detect the page
warped_page, debug_img = detect_page(cv_img)
if warped_page is None:
    print("Page detection failed.")
    raise SystemExit

# Detect paragraphs in the warped page
paragraphs, vis_img, dilated_img = detect_paragraphs(warped_page)

print("Processing completed successfully.")
