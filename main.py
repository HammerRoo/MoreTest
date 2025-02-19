import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
roi_folder = os.path.join(output_folder, "roi")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(roi_folder, exist_ok=True)

def save_image(image, name):
    path = os.path.join(output_folder, name)
    cv2.imwrite(path, image)
    print(f"Изображение сохранено: {path}")

def save_roi(image, name):
    path = os.path.join(roi_folder, name)
    cv2.imwrite(path, image)
    print(f"ROI сохранен: {path}")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(gray, "1_gray.png")

    blurred = cv2.GaussianBlur(gray, (11, 11), 7)
    save_image(blurred, "2_blurred.png")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2) # 11,2 стандарт, 19,3 также хороший результат
    save_image(thresh, "3_thresh.png")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_image(opened, "4_opened.png")

    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3) # на 3 видно рамкой весь номер, на 2 по отдельности цифры
    save_image(closed, "5_closed.png")

    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(closed, dilated_kernel, iterations=4)
    save_image(dilated, "6_dilated.png")

    return dilated

def find_and_draw_digits(image, processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_image = image.copy()
    detected_numbers = []

    for i, contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area > 100:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            roi = image[y:y+h, x:x+w]
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            equalized_roi = cv2.equalizeHist(gray_roi)

            blurred_roi = cv2.GaussianBlur(equalized_roi, (3, 3), 0)
            save_roi(blurred_roi, f"{i}_blurr_roi.png")

            binary_roi = cv2.adaptiveThreshold(equalized_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 5)
            save_roi(binary_roi, f"{i}_thresh_roi.png")

            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel_close, iterations=2) #1
            save_roi(closed_roi, f"{i}_morph_roi.png")

            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            eroded_roi = cv2.erode(closed_roi, kernel_erode, iterations=1)
            save_roi(eroded_roi, f"{i}_eroded_roi.png")
            
            scale_factor = 2
            resized_roi = cv2.resize(eroded_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            preprocessed_roi = resized_roi

            text = pytesseract.image_to_string(preprocessed_roi, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789')
            text = text.strip()
            
            if len(text) >= 3 and text.isdigit():
                detected_numbers.append(text)
    
    return output_image, detected_numbers

def process_video():
    video_path = input("Введите путь к видео: ")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    frame_skip = 5
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения.")
            break
        
        frame_count += 1
        
        cv2.imshow("Video", frame)
        
        if frame_count % frame_skip == 0:
            processed_image = preprocess_image(frame)
            _, detected_numbers = find_and_draw_digits(frame, processed_image)
            
            if detected_numbers:
                print(f"Кадр {frame_count}: Номер вагона виден: {detected_numbers}")
            else:
                print(f"Кадр {frame_count}: Номер вагона не виден")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_image():
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
        
        result_image, detected_numbers = find_and_draw_digits(image, processed_image)
        
        save_image(result_image, "7_result.jpg")

        if detected_numbers:
            print("Номер вагона виден: ", detected_numbers)
        else:
            print("Номер вагона не виден")

def main():
    while True:
        print("Выберите режим:")
        print("1. Обработка изображения")
        print("2. Обработка видео")
        print("3. Выход")
        choice = input("Введите номер режима: ")
        
        if choice == '1':
            process_image()
        elif choice == '2':
            process_video()
        elif choice == '3':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
            
if __name__ == "__main__":
    main()