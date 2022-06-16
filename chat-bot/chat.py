import logging
import torch
from mcn.model import CompatModel
import random

from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

from segmentation import process_photo
from complementary_only_score import score_outfit


# Enable logging
logging.basicConfig(
    # filename= 'telgramBot.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Main interactions
CHOOSING = range(1)

reply_keyboard = [
    ['Done'],
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False)


def model_compat():
    model = CompatModel(embed_size=1000, need_rep=True, vocabulary=2757)
    # Load pretrained weights
    weights_path = 'model_train_relation_vse_type_cond_scales.pth'
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def start(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: Start")

    context.user_data['chat_id'] = update.message.chat_id

    update.message.reply_text(
        "Hi! I am your style adviser bot. What are you wearing now?"
        "You can send me an image and I'll give you my opinion :)",
        reply_markup=markup
    )
    return CHOOSING


def outfit_query(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('infer_image.png')
    logger.info("Photo of %s: %s", user.first_name, 'infer_image.jpg')
    update.message.reply_text(
        'Thanks! The photo is being processed'
    )

    user_data = context.user_data

    # Infer image prediction
    result = process_photo('infer_image.png')

    score = score_outfit(result, model_compat())

    with open('index.txt') as file:
        counter = int(file.readline())

    update.message.reply_text(
        f"You get score {score} for your outlook!",
        reply_markup=markup
    )
    counter += 1
    with open('index.txt', 'w') as file:
        file.write(str(counter))
    return score


def done(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: done")

    user_data = context.user_data
    if 'user_outfit' in user_data:
        del user_data['user_outfit']

    update.message.reply_text(
        f"You look good! Bye bye until next time!"
    )

    user_data.clear()
    return ConversationHandler.END


def main() -> None:
    bot_token = '5538106353:AAG-X976RluMp7pDwsHys9Ci3phYHkNt1fg'
    print(bot_token)
    updater = Updater(bot_token, use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING: [
                MessageHandler(Filters.photo,
                               outfit_query),
            ],
        },
        fallbacks=[MessageHandler(Filters.regex('^Done$'), done)],
        per_message=False,
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()

