from flask import Flask, request, abort
from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from fintools import get_response_from_agent

app = Flask(__name__)

# ---- Replace with your real tokens ----
CHANNEL_ACCESS_TOKEN = "f7x88a8bNjsChLXDmhQE8lflFQYQIVIEboiN67X9mvJ2LcbRNgVDKYtiBhU8Dl2e4F9gJX+UyGJ1A61gxPoa81glhai/ExinG374v5BunqkhjZxL1joS7Q9wZW4p4a3NE7V/18mjsAIYO0aPFpUsywdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "309e5a7cec0f12143ba646ca6edae145"
# ---------------------------------------

# Create Configuration with your access token
config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

# Wrap it in ApiClient (required by v3 SDK)
api_client = ApiClient(config)

# Create MessagingApi using ApiClient
line_bot_api = MessagingApi(api_client)

# Webhook handler remains the same
handler = WebhookHandler(CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    print("Webhook received:", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_text = event.message.text

    # Reply using v3 MessagingApi
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=get_response_from_agent(user_text))]
        )
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)