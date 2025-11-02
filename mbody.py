import cv2
import pyttsx3
import time
import os
from datetime import datetime

# Инициализация движка для синтеза речи
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Скорость речи
engine.setProperty('voice', 'russian')  # Если доступен русский голос, иначе используйте дефолтный

# Загрузка каскадов для детекции лица (фронтальное и профиль)
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Инициализация веб-камеры Microsoft HD CAM (индекс 1; если не сработает, попробуйте 2)
cap = cv2.VideoCapture(1)

# Флаги и таймеры для подтверждения обнаружения
face_detected = False
first_detection_time = 0
last_detection_time = 0
confirmation_delay = 3  # Секунды для подтверждения
cooldown = 10  # Секунды между срабатываниями (увеличено для избежания спама)
last_alert_time = 0

# Папка для скриншотов
screenshots_dir = 'screenshots'
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)

print("Камера Microsoft HD CAM запущена. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра. Проверьте подключение камеры.")
        break

    # Преобразование в grayscale для детекции
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция фронтальных и профильных лиц
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, 1.1, 4)
    faces_profile = face_cascade_profile.detectMultiScale(gray, 1.1, 4)
    
    # Объединение результатов (все лица)
    all_faces = list(faces_frontal) + list(faces_profile)
    
    current_time = time.time()

    if len(all_faces) > 0:
        # Рисуем прямоугольники вокруг обнаруженных лиц (синий для фронтальных, красный для профильных)
        for (x, y, w, h) in faces_frontal:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in faces_profile:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        if not face_detected:
            # Первое обнаружение
            first_detection_time = current_time
            face_detected = True
        
        last_detection_time = current_time
        
        # Проверяем подтверждение (лицо в кадре стабильно 3 секунды)
        if (current_time - first_detection_time >= confirmation_delay) and \
           (current_time - last_alert_time >= cooldown):
            # Срабатывание: голос + скриншот
            engine.say("Обнаружен нарушитель")
            engine.runAndWait()
            print("Подтверждено: Обнаружено лицо нарушителя!")
            
            # Сохранение скриншота с timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f'intruder_{timestamp}.jpg')
            cv2.imwrite(screenshot_path, frame)
            print(f"Скриншот сохранен: {screenshot_path}")
            
            last_alert_time = current_time
            first_detection_time = current_time  # Сброс для следующего цикла
    
    else:
        # Лицо исчезло - сброс
        face_detected = False
        first_detection_time = 0
        last_detection_time = 0

    # Отображение кадра
    cv2.imshow('Детекция лица нарушителя', frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
engine.stop()