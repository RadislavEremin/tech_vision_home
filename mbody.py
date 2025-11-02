import cv2
import pyttsx3
import time
import os
from datetime import datetime
from ultralytics import YOLO  # Для детекции объектов (кошек) с YOLOv8

# Инициализация движка для синтеза речи
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Скорость речи
engine.setProperty('voice', 'russian')  # Если доступен русский голос, иначе используйте дефолтный

# Загрузка каскадов для детекции лица (фронтальное и профиль)
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Загрузка модели YOLOv8 для детекции объектов (включая кошек, класс 'cat' = 15)
model = YOLO('yolov8n.pt')  # Nano-модель, быстрая; скачается автоматически при первом запуске

# Инициализация веб-камеры Microsoft HD CAM (индекс 1; если не сработает, попробуйте 2)
cap = cv2.VideoCapture(1)

# Флаги и таймеры для подтверждения обнаружения (отдельно для лиц и кошек)
face_detected = False
cat_detected = False
face_first_time = 0
cat_first_time = 0
face_last_time = 0
cat_last_time = 0
confirmation_delay = 3  # Секунды для подтверждения
cooldown = 10  # Секунды между срабатываниями
face_last_alert = 0
cat_last_alert = 0

# Папка для скриншотов
screenshots_dir = 'screenshots'
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)

print("Камера Microsoft HD CAM запущена. Нажмите 'q' для выхода.")
print("ОТЛАДКА: Если кошка не детектируется, проверьте консоль на 'DET: Cats: N' (N > 0 значит видит кошку).")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра. Проверьте подключение камеры.")
        break

    # Преобразование в grayscale для детекции лиц
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция фронтальных и профильных лиц
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, 1.1, 4)
    faces_profile = face_cascade_profile.detectMultiScale(gray, 1.1, 4)
    all_faces = list(faces_frontal) + list(faces_profile)
    
    # Детекция объектов с YOLO (включая кошек)
    results = model(frame, verbose=False)  # Детекция на кадре
    cats_detected = []  # Список bbox для кошек
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls) == 15:  # Класс 15 = 'cat' в COCO датасете
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cats_detected.append((x1, y1, x2 - x1, y2 - y1))
    
    current_time = time.time()

    # ОТЛАДКА: Вывод количества детектированных кошек (каждые 30 кадров, чтобы не спамить)
    if int(current_time * 30) % 30 == 0:  # Примерно каждую секунду при 30 FPS
        print(f"ОТЛАДКА: DET: Cats: {len(cats_detected)} | Faces: {len(all_faces)}")

    # Логика для лиц (как раньше)
    if len(all_faces) > 0:
        # Рисуем прямоугольники вокруг лиц (синий для фронтальных, красный для профильных)
        for (x, y, w, h) in faces_frontal:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in faces_profile:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        if not face_detected:
            face_first_time = current_time
            face_detected = True
        face_last_time = current_time
        
        # Подтверждение для лица
        if (current_time - face_first_time >= confirmation_delay) and \
           (current_time - face_last_alert >= cooldown):
            engine.say("Обнаружен нарушитель")
            engine.runAndWait()
            print("Подтверждено: Обнаружено лицо нарушителя!")
            
            # Скриншот для лица
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f'intruder_{timestamp}.jpg')
            cv2.imwrite(screenshot_path, frame)
            print(f"Скриншот сохранен: {screenshot_path}")
            
            face_last_alert = current_time
            face_first_time = current_time  # Сброс для следующего
    
    else:
        face_detected = False
        face_first_time = 0
        face_last_time = 0

    # Логика для кошек
    if len(cats_detected) > 0:
        # Рисуем зеленые прямоугольники вокруг кошек
        for (x, y, w, h) in cats_detected:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Cat', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if not cat_detected:
            cat_first_time = current_time
            cat_detected = True
        cat_last_time = current_time
        
        # Подтверждение для кошки
        if (current_time - cat_first_time >= confirmation_delay) and \
           (current_time - cat_last_alert >= cooldown):
            engine.say("Обнаружена глупая кошка")
            engine.runAndWait()
            print("Подтверждено: Обнаружена глупая кошка!")
            
            # Скриншот для кошки
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f'cat_{timestamp}.jpg')
            cv2.imwrite(screenshot_path, frame)
            print(f"Скриншот сохранен: {screenshot_path}")
            
            cat_last_alert = current_time
            cat_first_time = current_time  # Сброс для следующего
    
    else:
        cat_detected = False
        cat_first_time = 0
        cat_last_time = 0

    # Отображение кадра
    cv2.imshow('Детекция лица/кошки', frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
engine.stop()