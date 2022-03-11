# importing all required libraries
import telebot
import yaml
from central_repo.notification import BASE_DIR
from telethon import TelegramClient, events, sync
from telethon.sync import TelegramClient
from telethon.tl.types import InputPeerChannel, InputPeerUser

config_file_path = BASE_DIR + '/configure.yml'
with open(config_file_path, 'r') as config_file:
  config = yaml.safe_load(config_file)
tel_config = config['TELEGRAM']

# get your api_id, api_hash, token
# from telegram as described above
api_id = tel_config["API_ID"]
api_hash = tel_config["API_HASH"]
token = tel_config['TOKEN']
message = "<h1>Message from telegram</h1><p>In html biro</p>"

# your phone number
phone = "+917017886892"

def send_message(
                 message,
                 reciever_id = 1706764213,
                 sender_phone = phone,
                 reciever_hash = 0,
                 api_id = api_id,
                 api_hash = api_hash,
                 token = token,
                 ):
  '''
  For sending messages with telegram
  message : The message which is to be sent to the reciever
  reciever_id : id of the reciever to 
  reicever_hash - The hash of the reciever
  
  '''
  # creating a telegram session and assigning
  # it to a variable client
  client = TelegramClient('session', api_id, api_hash)

  # connecting and building the session
  client.connect()

  # in case of script ran first time it will
  # ask either to input token or otp sent to
  # number or sent or your telegram id
  if not client.is_user_authorized():

    client.send_code_request(phone)
    
    # signing in the client
    client.sign_in(phone, input('Enter the code: '))


  try:
    # receiver reciever_id and access_hash, use
    # my reciever_id and access_hash for reference
    receiver = InputPeerUser(reciever_id, reciever_hash)

    # sending message using telegram client
    client.send_message(receiver, message, parse_mode='html')
  except Exception as e:
    
    # there may be many error coming in while like peer
    # error, wrong access_hash, flood_error, etc
    print(e);

  # disconnecting the telegram session
  client.disconnect()
