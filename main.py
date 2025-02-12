import cv2
import numpy as np
import pytesseract
import imutils
import os

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

while True:
    image_path = input("Введите путь к изображению (или 'exit' для выхода): ")
    if image_path.lower() == 'exit':
        break

    if not os.path.isfile(image_path):
        print("Файл не найден!")
        continue

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Не удалось загрузить изображение!")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "gray_image.png"), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(output_folder, "gaussian.png"), blurred)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(output_folder, "adaptive.png"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if w * h > 100:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_image_path = os.path.join(output_folder, "detected_digits.png")
    cv2.imwrite(output_image_path, img)
    print(f"Изображение с выделенными цифрами сохранено в {output_image_path}")

print("Программа завершена")