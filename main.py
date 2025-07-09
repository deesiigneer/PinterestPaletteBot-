import os
import hashlib
import time
import logging
import asyncio
import colorsys
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from aiohttp import ClientSession
from telegram import Bot, InputMediaPhoto
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

# --- Конфигурация ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
THEMES = [
    "aesthetic",
    "aesthetic cats",
    "aesthetic cars",
    "aesthetic girls",
    "aesthetic anime"
]
SAVE_DIR = "pinterest_downloads"
PUBLISHED_HASHES_FILE = "published_hashes.txt"
MAX_RETRIES = 10  # Максимальное количество попыток поиска изображения для темы


# --- Асинхронная инициализация Selenium ---
def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(options=chrome_options)
    return driver


async def load_published_hashes():
    try:
        if not os.path.exists(PUBLISHED_HASHES_FILE):
            return set()
        with open(PUBLISHED_HASHES_FILE, "r") as f:
            return set(line.strip() for line in f)
    except Exception as e:
        logging.error(f"Ошибка загрузки хэшей: {e}")
        return set()


async def save_hash(image_hash):
    try:
        with open(PUBLISHED_HASHES_FILE, "a") as f:
            f.write(f"{image_hash}\n")
    except Exception as e:
        logging.error(f"Ошибка сохранения хэша: {e}")


async def download_image(url, save_path):
    async with ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    with open(save_path, 'wb') as f:
                        f.write(await response.read())
                    return True
                return False
        except Exception as e:
            logging.error(f"Ошибка загрузки изображения {url}: {e}")
            return False


async def send_album_to_telegram(image_paths, caption):
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        media = [InputMediaPhoto(open(img, 'rb')) for img in image_paths]
        await bot.send_media_group(chat_id=CHANNEL_ID, media=media, caption=caption, parse_mode='MarkdownV2')
        return True
    except Exception as e:
        logging.error(f"Ошибка отправки в Telegram: {e}")
        return False


def convert_to_original_url(url):
    """Конвертирует URL изображения в оригинальное качество"""
    if "i.pinimg.com" in url:
        # Заменяем размер на originals
        parts = url.split('/')
        for i, part in enumerate(parts):
            if part in ['236x', '474x', '564x', '736x']:
                parts[i] = 'originals'
                break
        return '/'.join(parts)
    return url


async def get_pins_with_selenium(theme, retry_count=0):
    def fetch():
        driver = init_driver()
        try:
            search_query = theme.replace(" ", "%20")
            url = f"https://www.pinterest.com/search/pins/?q={search_query}"
            driver.get(url)
            time.sleep(1.5)

            # Прокрутка страницы для загрузки большего количества пинов
            for _ in range(2):
                driver.execute_script("window.scrollBy(0, 1000)")
                time.sleep(0.5)

            # Более точный поиск элементов
            pins = []
            pin_elements = driver.find_elements(By.CSS_SELECTOR, "div[data-test-id='pin'] img")

            for element in pin_elements:
                src = element.get_attribute("src") or element.get_attribute("data-src")
                if src and "i.pinimg.com" in src:
                    original_url = convert_to_original_url(src)
                    pins.append(original_url)
                    if len(pins) >= 10:  # Собираем больше URL для выбора
                        break

            return list(set(pins))  # Удаляем дубликаты
        except Exception as e:
            logging.error(f"Ошибка при парсинге темы {theme}: {e}")
            return []
        finally:
            driver.quit()

    loop = asyncio.get_event_loop()
    pins = await loop.run_in_executor(None, fetch)

    # Если не нашли достаточно пинов, пробуем еще раз
    if len(pins) < 5 and retry_count < MAX_RETRIES:
        logging.warning(f"Повторная попытка ({retry_count + 1}) для темы: {theme}")
        return await get_pins_with_selenium(theme, retry_count + 1)

    return pins


async def process_theme(theme, published_hashes):
    logging.info(f"Поиск по теме: {theme}")
    pins = await get_pins_with_selenium(theme)

    if not pins:
        logging.error(f"Не удалось найти пины для темы: {theme} после {MAX_RETRIES} попыток")
        return None

    # Пытаемся найти первое непубликовавшееся изображение
    for img_url in pins:
        img_hash = hashlib.md5(img_url.encode()).hexdigest()
        if img_hash in published_hashes:
            logging.info(f"Изображение уже публиковалось: {img_url}")
            continue

        save_path = os.path.join(SAVE_DIR, f"{theme.replace(' ', '_')}_{img_hash[:8]}.jpg")
        if await download_image(img_url, save_path):
            logging.info(f"Успешно загружено: {theme} -> {img_url}")
            return {'image': save_path, 'hash': img_hash}

    # Если все изображения уже публиковались, берем самое новое
    if pins:
        img_url = pins[0]
        img_hash = hashlib.md5(img_url.encode()).hexdigest()
        save_path = os.path.join(SAVE_DIR, f"{theme.replace(' ', '_')}_{img_hash[:8]}.jpg")
        if await download_image(img_url, save_path):
            logging.warning(f"Используем уже публиковавшееся изображение для темы: {theme}")
            return {'image': save_path, 'hash': img_hash}

    return None


def get_dominant_color(image_path, k=3):
    """Определяет доминирующий цвет изображения с помощью K-mean clustering"""
    image = Image.open(image_path)
    image = image.resize((150, 150))  # Уменьшаем для ускорения обработки
    np_image = np.array(image)
    np_image = np_image.reshape((np_image.shape[0] * np_image.shape[1], 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(np_image)

    # Получаем RGB доминирующего цвета
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return tuple(dominant_color.astype(int))


def colors_are_similar(color1, color2, threshold=50):
    """Проверяет схожесть двух цветов по евклидову расстоянию"""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    distance = np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
    return distance < threshold


async def process_theme(theme, published_hashes, base_color=None):
    logging.info(f"Поиск по теме: {theme}")
    pins = await get_pins_with_selenium(theme)

    if not pins:
        logging.error(f"Не удалось найти пины для темы: {theme} после {MAX_RETRIES} попыток")
        return None

    # Пытаемся найти изображение, подходящее по цвету
    for img_url in pins:
        img_hash = hashlib.md5(img_url.encode()).hexdigest()
        if img_hash in published_hashes:
            continue

        save_path = os.path.join(SAVE_DIR, f"{theme.replace(' ', '_')}_{img_hash[:8]}.jpg")
        if await download_image(img_url, save_path):
            try:
                current_color = get_dominant_color(save_path)

                # Если базовый цвет не задан (первое изображение) или цвета схожи
                if base_color is None or colors_are_similar(current_color, base_color):
                    logging.info(f"Успешно загружено: {theme} -> {img_url}")
                    return {
                        'image': save_path,
                        'hash': img_hash,
                        'color': current_color
                    }
                else:
                    logging.info(f"Изображение не подходит по цветовой гамме: {img_url}")
                    os.remove(save_path)  # Удаляем неподходящее изображение
            except Exception as e:
                logging.error(f"Ошибка анализа цвета: {e}")
                continue

    # Если не нашли подходящее по цвету, берем первое доступное
    for img_url in pins:
        img_hash = hashlib.md5(img_url.encode()).hexdigest()
        save_path = os.path.join(SAVE_DIR, f"{theme.replace(' ', '_')}_{img_hash[:8]}.jpg")
        if await download_image(img_url, save_path):
            current_color = get_dominant_color(save_path)
            logging.warning(f"Используем изображение без учета цвета для темы: {theme}")
            return {
                'image': save_path,
                'hash': img_hash,
                'color': current_color
            }

    return None


async def main():
    published_hashes = await load_published_hashes()
    downloaded_files = []
    new_hashes = []
    base_color = None

    try:
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Сначала собираем все изображения
        results = []
        for theme in THEMES:
            result = None
            attempts = 0

            while not result and attempts < MAX_RETRIES:
                result = await process_theme(theme, published_hashes, base_color)
                attempts += 1
                if not result:
                    await asyncio.sleep(2)

            if result:
                results.append(result)
                # Обновляем базовый цвет по первому удачному изображению
                if base_color is None and len(results) > 0:
                    base_color = results[0]['color']
            else:
                logging.error(f"Не удалось найти изображение для темы {theme}")

        # Если нашли все 5 изображений
        if len(results) == 5:
            # Сортируем по схожести с базовым цветом
            results.sort(key=lambda x: np.sqrt(
                sum((a - b) ** 2 for a, b in zip(x['color'], base_color))
            ) if base_color else 0)

            downloaded_files = [r['image'] for r in results]
            new_hashes = [r['hash'] for r in results]

            if await send_album_to_telegram(downloaded_files, f'[{os.getenv("CHANNEL_NAME")}]({os.getenv("CHANNEL_URL")})'):
                for img_hash in new_hashes:
                    await save_hash(img_hash)
                logging.info(f"Успешно опубликовано 5 изображений в одной цветовой гамме")

                # Логируем доминирующие цвета
                for i, result in enumerate(results):
                    logging.info(f"Изображение {i + 1}: RGB {result['color']}")

    except Exception as e:
        logging.error(f"Критическая ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    logging.info("=== Запуск бота с цветовой фильтрацией ===")
    asyncio.run(main())