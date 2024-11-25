from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, CallbackContext
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import io
import nest_asyncio
import torch
import numpy as np
import cv2
from fastai.vision.all import *
from PIL import Image

def generate_gradcam(learn, image, target_layer_name="0"):
    """Генерация Grad-CAM для заданной модели и изображения."""
    model = learn.model.eval()
    
    # Находим целевой слой
    target_layer = dict(model.named_modules())[target_layer_name]
    activations = []
    gradients = []

    # Hook для активаций
    def forward_hook(module, input, output):
        activations.append(output)

    # Hook для градиентов
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Устанавливаем хуки
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Подготавливаем изображение
    dl = learn.dls.test_dl([image])  # Создаём DataLoader
    for batch in dl:
        image_tensor = batch[0]  # Получаем входное изображение в формате тензора

    # Пропускаем изображение через модель
    output = model(image_tensor)
    pred_idx = output.argmax(dim=1).item()
    
    # Вычисляем градиенты
    model.zero_grad()
    output[0, pred_idx].backward()

    # Получаем активации и усреднённые градиенты
    activations = activations[0].squeeze().detach()
    gradients = gradients[0].squeeze().detach().mean(dim=(1, 2))

    # Генерируем карту важности
    gradcam_map = (activations * gradients.view(-1, 1, 1)).sum(0)
    gradcam_map = torch.clamp(gradcam_map, min=0)
    gradcam_map = gradcam_map / gradcam_map.max()

    # Убираем хуки
    forward_handle.remove()
    backward_handle.remove()

    return gradcam_map.numpy()



def overlay_gradcam(image, gradcam_map):
    """Наложение тепловой карты Grad-CAM на изображение."""
    gradcam_map = cv2.resize(gradcam_map, (image.size[0], image.size[1]))
    gradcam_map = (gradcam_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gradcam_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.array(image) * 0.5 + heatmap * 0.5
    return Image.fromarray(overlay.astype(np.uint8))


# Apply async support
nest_asyncio.apply()

# Load the models
learn_brain = load_learner('brain_model.pkl')
learn_skin = load_learner('eye_model.pkl')

# Language configurations
LANGUAGES = {
    "ru": {
        "start": "Привет! Выберите язык для работы:",
        "language_set": 'Вы выбрали - {model_name}',
        "help": "Команды:\n/start - Начать работу\n/help - Список команд",
        "analyze_result": "Результат анализа: Класс: {pred}, вероятность: {prob:.2f}",
        "model_select": "🤖 Добро пожаловать! Этот бот использует искусственный интеллект для анализа снимков МРТ головного мозга и помогает выявить возможные признаки рака. 🚨\n\n"
                       "📸 Отправьте снимок, и бот:\n"
                       "1️⃣ Проанализирует изображение с помощью нейросети.\n"
                       "2️⃣ Покажет, есть ли подозрения на наличие болезни.\n\n"
                       "🧑‍⚕️ Важно: этот инструмент предназначен только для информационных целей. Для точного диагноза обратитесь к врачу!\n\n"
                       'Если у вас есть особые пожелания или функции, которые нужно отразить в описании, сообщите мне @eralyf!\n'
                       'Выберите модель для анализа:',
        "brain_model": "Модель для анализа МРТ головного мозга",
        "eye_model": "Модель для анализа глазных заболеваний",
    },
    "kk": {
        "start": "Сәлем! Жұмыс істеу үшін тілді таңдаңыз:",
        "language_set": 'Сіз таңдаған модель - {model_name}',
        "help": "Бұйрықтар:\n/start - Бастау\n/help - Бұйрықтар тізімі",
        "analyze_result": "Талдау нәтижесі: Сынып: {pred}, ықтималдық: {prob:.2f}",
        "model_select": "🤖 Қош келдіңіз! Бұл бот жасанды интеллектті пайдаланып, бас миының МРТ суреттерін талдайды және қатерлі ісіктің ықтимал белгілерін анықтауға көмектеседі. 🚨\n\n"
                       "📸 Суретті жіберіңіз, бот:\n"
                       "1️⃣ Суретті нейрондық желі арқылы талдайды.\n"
                       "2️⃣ Ауыруға күдік бар-жоғын көрсетеді.\n\n"
                       "🧑‍⚕️ Маңызды: бұл құрал тек ақпараттық мақсаттарға арналған. Дәл диагноз алу үшін дәрігерге барыңыз!\n\n"
                       "Егер сізде ерекше тілектер немесе сипаттамаға енгізілетін функциялар болса, маған хабарлаңыз @eralyf!"
                       "Модельді таңдаңыз:",
        "brain_model": "МРТ бас миы моделін талдау",
        "eye_model": "Көз ауруларын талдайтын модель",
    },
}

# Store user language and model preferences
user_languages = {}
user_models = {}

# Get language for a user
def get_language(update: Update) -> str:
    user_id = update.effective_user.id
    return user_languages.get(user_id, "ru")

# Get model for a user
def get_model(update: Update) -> str:
    user_id = update.effective_user.id
    return user_models.get(user_id, "brain_model")

# Command /start
async def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("Русский", callback_data="set_language_ru")],
        [InlineKeyboardButton("Қазақша", callback_data="set_language_kk")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(LANGUAGES["ru"]["start"], reply_markup=reply_markup)

# Handler to set language
async def set_language(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    if query.data == "set_language_ru":
        user_languages[query.from_user.id] = "ru"
        lang = "ru"
    elif query.data == "set_language_kk":
        user_languages[query.from_user.id] = "kk"
        lang = "kk"

    keyboard = [
        [InlineKeyboardButton(LANGUAGES[lang]["brain_model"], callback_data="set_model_brain")],
        [InlineKeyboardButton(LANGUAGES[lang]["eye_model"], callback_data="set_model_eye")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(text=LANGUAGES[lang]["model_select"], reply_markup=reply_markup)

# Handler to set model
async def set_model(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    if query.data == "set_model_brain":
        user_models[query.from_user.id] = "brain_model"
    elif query.data == "set_model_eye":
        user_models[query.from_user.id] = "eye_model"

    lang = get_language(update)

    model_name = LANGUAGES[lang]["brain_model"] if user_models[query.from_user.id] == "brain_model" else LANGUAGES[lang]["eye_model"]
    await query.edit_message_text(text=LANGUAGES[lang]["language_set"].format(model_name=model_name))

# Command /help
async def help_command(update: Update, context: CallbackContext):
    lang = get_language(update)
    await update.message.reply_text(LANGUAGES[lang]["help"])

# Image analysis handler
async def analyze_image(update: Update, context: CallbackContext):
    lang = get_language(update)
    model_choice = get_model(update)

    # Load image
    photo_file = await update.message.photo[-1].get_file()
    file_data = await photo_file.download_as_bytearray()
    image = Image.open(io.BytesIO(file_data)).convert("RGB")

    # Choose model based on user selection
    learn = learn_brain if model_choice == "brain_model" else learn_skin

    # Analyze image
    pred, pred_idx, probs = learn.predict(image)
    gradcam_map = generate_gradcam(learn, image)
    gradcam_image = overlay_gradcam(image, gradcam_map)

    # Сохранение Grad-CAM для отправки
    gradcam_buffer = io.BytesIO()
    gradcam_image.save(gradcam_buffer, format="PNG")
    gradcam_buffer.seek(0)

    result = LANGUAGES[lang]["analyze_result"].format(pred=pred, prob=probs[pred_idx])
    await update.message.reply_text(result)

    # Send Grad-CAM image
    await update.message.reply_photo(gradcam_buffer)

# Main bot loop
async def main():
    TOKEN = "7643217203:AAECM5T-AoJ67fEbT3EUtYQe5QNSFID2lmk"
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(set_language, pattern="^set_language_"))
    app.add_handler(CallbackQueryHandler(set_model, pattern="^set_model_"))
    app.add_handler(MessageHandler(filters.PHOTO, analyze_image))

    await app.run_polling()

# Run the bot
await main()
