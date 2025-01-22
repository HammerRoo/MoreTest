import cv2
import numpy as np
import pytesseract
import imutils
import os

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

image_path = input("Введите путь к изображению для распознавания текста: ")

try:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")
except Exception as e:
    print(f"Ошибка при загрузке изображения: {e}")
    exit()

cv2.imwrite(os.path.join(output_folder, "original_image.jpg"), img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_folder, "gray_image.jpg"), gray)

gray = cv2.bilateralFilter(gray, 13, 15, 15)
cv2.imwrite(os.path.join(output_folder, "bilateral_filtered.jpg"), gray)

# edged = cv2.Canny(gray, 50, 200)
# cv2.imwrite(os.path.join(output_folder, "edged_image.jpg"), edged)

# contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)

# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# possible_texts = []
# number_found = False
# for idx, c in enumerate(contours):
#     mask = np.zeros(gray.shape, np.uint8)
#     cv2.drawContours(mask, [c], -1, 255, -1)

#     (x, y, w, h) = cv2.boundingRect(c)
#     roi = gray[y:y+h, x:x+w]

#     if w > 50 and h > 15:
#         if not number_found:
#             cv2.imwrite(os.path.join(output_folder, "first_roi.jpg"), roi)

#         config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
#         text = pytesseract.image_to_string(roi, config=config)

#         numbers = "".join(filter(str.isdigit, text))
#         if len(numbers) >= 8:
#             possible_texts.append(numbers)
            
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(img, numbers, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#             number_found = True
#             break

# final_image_path = os.path.join(output_folder, "final_image_with_text.jpg")
# cv2.imwrite(final_image_path, img)

# if possible_texts:
#     print("\nРаспознанные номера:")
#     for num in possible_texts:
#         print(num)
# else:
#     print("Номера не найдены.")

# print(f"\nПромежуточные изображения сохранены в папке: {output_folder}")
# print(f"Финальное изображение сохранено как: {final_image_path}")