
# /usr/bin/env python
# Download the twilio-python library from twilio.com/docs/libraries/python
from twilio.rest import Client
import yaml

with open ('./configure.yml', 'r') as config:
  config = yaml.safe_load(config)

account_sid = config['SMS']['account_sid']
auth_token = config['SMS']['auth_token']
phone_no = config['SMS']['phone_no']

client = Client(account_sid, auth_token)

def send_sms(numbers, message):

  if not isinstance(message, str) : raise ValueError
  if len(message) == 0 : raise ValueError

  if len(numbers == 1):
    client.api.account.messages.create(
        to= numbers,
        from_= phone_no,
        body=message)
  else:
    for number in numbers:
      client.api.account.messages.create(
          to= number,
          from_= phone_no,
          body=message)
    