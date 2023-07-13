from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
import time

from aiogram.types import InputFile

start_time = None
from config import TOKEN
import requests
from aiogram import Bot, Dispatcher, executor, types

storage = MemoryStorage()

bot = Bot(token=TOKEN)

dp = Dispatcher(bot, storage=storage)


@dp.message_handler(commands=['start'], state="*")
async def start_handler(message: types.Message):
    await message.answer(
        'Добро пожаловать!\nЭто бот, позволяющий проверить свои навыки набора текста.\n\n/text - начать\n\n/rules - правила тренажера')


@dp.message_handler(commands=['rules'], state="*")
async def start_handler(message: types.Message):
    await message.answer(
        'Бот будет генерировать и давать вам случайные картинки. Вам, в свою очередь, необходимо переписать текст на этих картинках как можно быстрее и как можно точнее.\nПосле каждого написанного текста вам будет выводится статистика\n - Затраченное время\n - Сходство текстов\n - Скорость печати\nЕсли сходство текстов будет меньше 70%, то вам придется переписывать его.')


URL = "https://fish-text.ru/get"


@dp.message_handler(commands=['text'], state="*")
async def text_handler(message: types.Message, state: FSMContext):
    await message.answer('Старт!')
    text, very_start_time = set_text()
    await state.update_data(text=text, start_time=very_start_time)
    await state.update_data(very_start_time=very_start_time)
    photo = create_text_photo(text)
    await message.answer_photo(caption="Введите то, что на изображении", photo=photo)
    photo.close()
    await state.set_state("enter_text")


def set_text():
    response = requests.get(URL)
    response_dict = response.json()
    text = response_dict["text"]
    start_time = time.time()
    return text, start_time

def levenshtein_distance(text1, text2):
    m = len(text1)
    n = len(text2)

    # Создаем матрицу размером (m+1) x (n+1) и заполняем ее нулями
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Заполняем первую строку и первый столбец матрицы
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Вычисляем расстояние Левенштейна для каждой пары символов
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,  # Удаление символа
                               dp[i][j - 1] + 1,  # Вставка символа
                               dp[i - 1][j - 1] + 1)  # Замена символа

    # Возвращаем значение расстояния Левенштейна для двух текстов
    return dp[m][n]

def compare_texts(text1, text2):
    distance = levenshtein_distance(text1, text2)
    max_length = max(len(text1), len(text2))
    similarity = 1 - (distance / max_length)
    return similarity

@dp.message_handler(state="enter_text")
async def text_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()
    text = data["text"]
    similarity = compare_texts(message.text, text) * 100
    end_time = time.time()
    start_time = data["start_time"]
    if similarity >= 70:
        delta = end_time - start_time
        word_count = len(text.split())
        typing_speed = 60 * word_count / delta
        await message.answer(f"Отлично, ты справился за {delta:.2f} секунд.")
        await message.answer(f"Сходство текста - {similarity:.2f}%")
        await message.answer(f"Скорость печати: {int(typing_speed)} слов в минуту")
        text, start_time = set_text()
        await state.update_data(text=text, start_time=start_time)
    else:
        await message.answer(f"В твоем тексте слишком много ошибок, сходство - {similarity:.2f}%!\nНеобходимо его перписать!")
        data = await state.get_data()
        text = data["text"]
        similarity = compare_texts(message.text, text) * 100
        end_time = time.time()
        start_time = data["start_time"]

    photo = create_text_photo(text)
    await message.answer_photo(caption="Введите то, что на изображении.", photo=photo)
    photo.close()


from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import textwrap


def create_text_photo(text, font_size=12):
    cnt_of_symbols_on_the_line = 70
    # Сделать переносы строк
    wrapped_lines = textwrap.wrap(text, cnt_of_symbols_on_the_line)
    wrapped_text = "\n".join(wrapped_lines)

    # Загрузка шрифта
    font = ImageFont.truetype('arial.ttf', font_size)

    # Создание изображения
    image = Image.new('RGB', (0, 0), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    (x0, y0, x1, y1) = draw.multiline_textbbox((5, 5), text=wrapped_text, font=font)
    new_width = x1 + 10
    new_height = y1 + 10

    image = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.multiline_text((5, 5), wrapped_text, font=font, fill=(0, 0, 0), align="left")

    photo = BytesIO()
    photo.name = 'image.jpeg'
    image.save(photo, 'JPEG')
    photo.seek(0)

    return photo


if __name__ == "__main__":
    # Запуск бота
    executor.start_polling(dispatcher=dp, skip_updates=True)
