from main import TelegramBot
import traceback


def train_model():
    ## do something ##
    results = {
        'train accuracy' : 'train_acc'
    }
    return 'lol'


TOKEN = '5538106353:AAG-X976RluMp7pDwsHys9Ci3phYHkNt1fg'
MYID = '700377217'

bot = TelegramBot(TOKEN, MYID)
results = ''
# Run this
try:
    results = train_model()

# If error occurs, send the error with its trace
except Exception as e:
    print(traceback.format_exc())
    bot.send_error_message(traceback.format_exc())

bot.send_message(results)
bot.send_image('test.png', caption='here is the image of the results')