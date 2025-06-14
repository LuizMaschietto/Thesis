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
    # Debug: draw points on blank image for visualization
    debug_img = np.zeros((300, 400, 3), dtype=np.uint8)
    for i, point in enumerate(rect):
        pos = tuple(point.astype(int))
        cv2.circle(debug_img, pos, 8, (0, 255, 0), -1)
        cv2.putText(debug_img, f"P{i}", (pos[0]+5, pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite("/sd/Debug_order_points.jpg", debug_img)
    print("Saved Debug_order_points.jpg showing ordered corner points.")
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

    # Debug: visualize source and destination points overlaid on images
    debug_src = image.copy()
    for i, point in enumerate(rect):
        pos = tuple(point.astype(int))
        cv2.circle(debug_src, pos, 10, (0, 255, 255), 3)
        cv2.putText(debug_src, f"S{i}", (pos[0]+5, pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite("/sd/Debug_four_point_src.jpg", debug_src)
    debug_dst = np.zeros((maxHeight, maxWidth, 3), dtype=np.uint8)
    for i, point in enumerate(dst):
        pos = tuple(point.astype(int))
        cv2.circle(debug_dst, pos, 10, (255, 0, 255), 3)
        cv2.putText(debug_dst, f"D{i}", (pos[0]+5, pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.imwrite("/sd/Debug_four_point_dst.jpg", debug_dst)
    print("Saved Debug_four_point_src.jpg and Debug_four_point_dst.jpg showing source and destination points.")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    cv2.imwrite("/sd/Debug_four_point_warped.jpg", warped)
    print("Saved Debug_four_point_warped.jpg showing persp. transformed image.")
    return warped

def detect_page(cv_img):
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/sd/Debug_page_gray.jpg", gray)
    print("Saved Debug_page_gray.jpg grayscale image.")

    # Blur to smooth noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("/sd/Debug_page_blur.jpg", blur)
    print("Saved Debug_page_blur.jpg blurred image.")

    # Adaptive threshold + invert to get white text on black
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    cv2.imwrite("/sd/Debug_page_thresh.jpg", thresh)
    print("Saved Debug_page_thresh.jpg adaptive threshold image.")

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found!")
        return None, None

    page_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
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
    cv2.drawContours(debug_img, [page_contour], -1, (0, 255, 0), 6)
    cv2.imwrite("/sd/Debug_page_contour.jpg", debug_img)
    print("Saved Debug_page_contour.jpg showing detected page contour.")

    # Perspective transform to get top-down view of page
    warped = four_point_transform(cv_img, page_contour.reshape(4, 2))

    return warped, debug_img


def remove_border(img):
    # Convert to grayscale if the image is not already
    if len(img.shape) == 3:  # Check if the image has color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]

    # Apply morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Get bounding box coordinates from the largest external contour
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # Check if any contours were found
        big_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(big_contour)

        # Crop the image to remove the border
        imgnoborder = img[y:y+h, x:x+w]
        cv2.imwrite("/sd/debug_remove_border.jpg", imgnoborder)

        return imgnoborder
    else:
        print("No contours found.")
        return img  # Return the original image if no contours are found




def detect_paragraphs(page_img):
    # Convert to grayscale and blur
   # gray_page = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(page_img, (7, 7), 0)
    
    # Thresholding (Otsu’s method, invert so text is white)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    cv2.imwrite("/sd/debug_Thresh.jpg", thresh)

    h, w = thresh.shape

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour touching border (probable border contour)
    border_contour = None
    max_area = 0
    border_threshold = 5  # pixels inside border to consider "touching"

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        touches_border = (
            x <= border_threshold or
            y <= border_threshold or
            (x + cw) >= (w - border_threshold) or
            (y + ch) >= (h - border_threshold)
        )
        if touches_border and area > max_area:
            max_area = area
            border_contour = cnt

    # Create white mask and paint black only the border contour area
    mask = np.ones_like(thresh, dtype=np.uint8) * 255  # white mask

    if border_contour is not None:
        cv2.drawContours(mask, [border_contour], -1, 0, thickness=cv2.FILLED)  # black inside border

    # Apply mask to threshold to remove border region only
    thresh_no_border = cv2.bitwise_and(thresh, thresh, mask=mask)
    cv2.imwrite("/sd/debug_Thresh_no_border.jpg", thresh_no_border)

    # Dilate to connect paragraph lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh_no_border, kernel, iterations=4)
    cv2.imwrite("/sd/debug_Dilated.jpg", dilated)

    # Find paragraph contours now on dilated image
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detected {len(contours)} paragraph contours after border removal")

    if not contours:
        print("No paragraphs detected!")
        return [], page_img.copy(), dilated

    # Sort contours top-to-bottom
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    paragraphs = []
    vis_img = page_img.copy()
    for i, cnt in enumerate(sorted_contours):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = cw / ch if ch != 0 else 0
        if area > 120 and aspect_ratio < 5:
            paragraph = page_img[y:y+ch, x:x+cw]
            save_path = f"/sd/paragraph_{i}.jpg"
            cv2.imwrite(save_path, paragraph)
            paragraphs.append(save_path)
            cv2.rectangle(vis_img, (x, y), (x+cw, y+ch), (0,255,0), 2)
            print(f"Paragraph {i} saved: {save_path}")

    cv2.imwrite("/sd/debug_paragraphs_detected.jpg", vis_img)
    print("Paragraph detection completed.")

    return paragraphs, vis_img, dilated





wait_for_button_press()

# Capture image
img = cam.read()
if img is None:
    print("Error capturing image.")
    raise SystemExit
img_path = "/sd/captured_page.jpg"
img.save(img_path)

#Debug
img_PageNTest="/sd/PagT.png"
img_PageWTest="/sd/PagTWarp.png"
img_ParagraphTest="/sd/ImagemTParagrafo.png"


print("Image captured and saved in:", img_path)

# Process the captured image
cv_img = cv2.imread(img_PageWTest)
if cv_img is None:
    print("Error reading image with OpenCV.")
    raise SystemExit

# Detect the page
warped_page, debug_img = detect_page(cv_img)
if warped_page is None:
    print("Page detection failed.")
    raise SystemExit

noborder= remove_border(warped_page)

# Detect paragraphs in the warped page
paragraphs, vis_img, dilated_img = detect_paragraphs(noborder)

print("Processing completed successfully.")
