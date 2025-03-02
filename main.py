import cv2
import numpy as np
import os
import pytesseract
from collections import OrderedDict

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
roi_folder = os.path.join(output_folder, "roi")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(roi_folder, exist_ok=True)

import cv2
import numpy as np

import cv2
import numpy as np

def correct_distortion(frame, strength=0.5):
    h, w = frame.shape[:2]
    
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            x = (j - w / 2) / (w / 2)
            y = (i - h / 2) / (h / 2)
            
            r = np.sqrt(x**2 + y**2)
            theta = 1.0 / (1.0 + strength * r**2)
            
            map_x[i, j] = (x * theta + 1) * (w / 2)
            map_y[i, j] = (y * theta + 1) * (h / 2)
    
    corrected_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return corrected_frame

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

    blurred = cv2.GaussianBlur(gray, (11, 11), 7) # 6
    save_image(blurred, "2_blurred.png")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    save_image(thresh, "3_thresh.png")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_image(opened, "4_opened.png")


    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) # 7, 7
    dilated = cv2.dilate(opened, dilated_kernel, iterations=4) # 4
    save_image(dilated, "5_dilated.png")

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    save_image(output_image, "6_result.jpg")

    return dilated

def find_and_draw_digits(image, processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_numbers = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        roi = image[y:y+h, x:x+w]
        text_clean = pytesseract.image_to_string(
            roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_clean) >= 4 and text_clean.isdigit():
            detected_numbers.append(text_clean)
            print(f"Номер найден на original_roi")
            save_roi(roi, f"1_{i}_original_roi.png")
            continue
        save_roi(roi, f"1_{i}_original_roi.png")

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_gray = pytesseract.image_to_string(
            roi_gray, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_gray) >= 4 and text_gray.isdigit():
            detected_numbers.append(text_gray)
            print(f"Номер найден на gray_roi")
            save_roi(roi_gray, f"2_{i}_gray_roi.png")
            continue
        save_roi(roi_gray, f"2_{i}_gray_roi.png")

        binary_roi = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
        text_binary = pytesseract.image_to_string(
            binary_roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_binary) >= 4 and text_binary.isdigit():
            detected_numbers.append(text_binary)
            print(f"Номер найден на binary_roi")
            save_roi(binary_roi, f"3_{i}_binary_roi.png")
            continue
        save_roi(binary_roi, f"3_{i}_binary_roi.png")

        # erosed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # erosed_roi = cv2.erode(binary_roi, erosed_kernel, iterations=1)
        # text_erose = pytesseract.image_to_string(
        #     erosed_roi, 
        #     config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        # ).strip()
        # if len(text_erose) >= 4 and text_erose.isdigit():
        #     detected_numbers.append(text_erose)
        #     print(f"Номер найден на erose_roi")
        #     save_roi(erosed_roi, f"4_{i}_erosed_roi.png")
        #     continue
        # save_roi(erosed_roi, f"4_{i}_erosed_roi.png")

        dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_roi = cv2.dilate(binary_roi, dilated_kernel, iterations=2)
        text_dilate = pytesseract.image_to_string(
            dilated_roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_dilate) >= 4 and text_dilate.isdigit():
            detected_numbers.append(text_dilate)
            print(f"Номер найден на dilate_roi")
            save_roi(dilated_roi, f"5_{i}_dilated_roi.png")
            continue
        save_roi(dilated_roi, f"5_{i}_dilated_roi.png")

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, close_kernel, iterations=5)
        text_close = pytesseract.image_to_string(
            morph_roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_close) >= 4 and text_close.isdigit():
            detected_numbers.append(text_close)
            print(f"Номер найден на close_roi")
            save_roi(morph_roi, f"6_{i}_morph_close_roi.png")
            continue
        save_roi(morph_roi, f"6_{i}_morph_close_roi.png")

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_OPEN, open_kernel, iterations=1)
        text_open = pytesseract.image_to_string(
            morph_roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_open) >= 4 and text_open.isdigit():
            detected_numbers.append(text_open)
            print(f"Номер найден на open_roi")
            save_roi(morph_roi, f"7_{i}_morph_open_roi_2.png")
            continue
        save_roi(morph_roi, f"7_{i}_morph_open_roi_2.png")

        # scale_factor = 2
        # resized_roi = cv2.resize(
        #     morph_roi,
        #     None,
        #     fx=scale_factor,
        #     fy=scale_factor,
        #     interpolation=cv2.INTER_CUBIC
        # )

        text_processed = pytesseract.image_to_string(
            morph_roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()

        if len(text_processed) >= 6 and text_processed.isdigit():
            detected_numbers.append(text_processed)
            print(f"Номер найден после предобработки")

    return list(OrderedDict.fromkeys(detected_numbers))

def process_video():
    video_path = input("Введите путь к видео: ")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    frame_skip = 10
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
            detected_numbers = find_and_draw_digits(frame, processed_image)
            
            if detected_numbers:
                print(f"Номер вагона виден")
            else:
                print(f"Номер вагона не виден")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_selected_frame():
    video_path = input("Введите путь к видео: ")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    frame_count = 0
    paused = False
    
    #distortion_strength = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения.")
            break
        
        frame_count += 1
        
        #corrected_frame = correct_distortion(frame, distortion_strength)

        cv2.imshow("Video", frame) #corrected_frame
        
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break

        elif key == ord(' '):
            processed_image = preprocess_image(frame) #corrected_frame
            detected_numbers = find_and_draw_digits(frame, processed_image) #corrected_frame
            
            if detected_numbers:
                print(f"Номер вагона виден")
            else:
                print(f"Номер вагона не виден")

        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Видео на паузе. Нажмите 'p', чтобы продолжить.")
            else:
                print("Воспроизведение возобновлено.")
                
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
        
        detected_numbers = find_and_draw_digits(image, processed_image)

        if detected_numbers:
            print("Номер вагона виден")
        else:
            print("Номер вагона не виден")

def main():
    while True:
        print("Выберите режим:")
        print("1. Обработка изображения")
        print("2. Обработка видео")
        print("3. Запуск видео и обработка выбранных кадров.")
        print("4. Выход")
        choice = input("Введите номер режима: ")
        
        if choice == '1':
            process_image()
        elif choice == '2':
            process_video()
        elif choice == '3':
            process_selected_frame()
        elif choice == '4':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
            
if __name__ == "__main__":
    main()