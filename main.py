import cv2
import numpy as np
import os

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

def save_image(image, name):
    path = os.path.join(output_folder, name)
    cv2.imwrite(path, image)
    print(f"Изображение сохранено: {path}")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(gray, "1_gray.jpg")

    blurred = cv2.GaussianBlur(gray, (9, 9), 3)
    save_image(blurred, "2_blurred.jpg")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2) # 11,2 стандарт, 19,3 также хороший результат
    save_image(thresh, "3_thresh.jpg")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_image(opened, "4_opened.jpg")

    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3) # на 3 видно рамкой весь номер, на 2 по отдельности цифры
    save_image(closed, "5_closed.jpg")

    return closed

def find_and_draw_digits(image, processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_image = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 100:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return output_image

def main():
    while True:
        image_path = input("Введите путь к изображению (или 'exit' для выхода): ")
        
        if image_path.lower() == 'exit':
            print("Выход из программы.")
            break
        
        image = cv2.imread(image_path)
        
        if image is None:
            print("Ошибка загрузки изображения. Проверьте путь и попробуйте снова.")
            continue
        
        processed_image = preprocess_image(image)
        
        result_image = find_and_draw_digits(image, processed_image)
        
        save_image(result_image, "6_result.jpg")

if __name__ == "__main__":
    main()