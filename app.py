from flask import Flask, request, jsonify, abort
import threading
import uuid
from datetime import datetime, timezone

from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from fintools import get_response_from_agent

app = Flask(__name__)

# ---- IMPORTANT: rotate these secrets; don't hardcode in real deployments ----
# ---- Replace with your real tokens ----
CHANNEL_ACCESS_TOKEN = "f7x88a8bNjsChLXDmhQE8lflFQYQIVIEboiN67X9mvJ2LcbRNgVDKYtiBhU8Dl2e4F9gJX+UyGJ1A61gxPoa81glhai/ExinG374v5BunqkhjZxL1joS7Q9wZW4p4a3NE7V/18mjsAIYO0aPFpUsywdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "309e5a7cec0f12143ba646ca6edae145"

config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(config)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

DEFAULT_USER_ID = "U192772f59a4321d51d8b084fde86748d"

# In-memory job status (resets on restart). For production use Redis/DB.
JOB_STATUS = {}


@app.get("/")
def health():
    return "OK", 200


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

"""
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_text = event.message.text
    reply = get_response_from_agent(user_text)

    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply)]
        )
    )
"""


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_text = event.message.text
    reply_token = event.reply_token

    # Reply to LINE immediately to prevent timeout
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text="⏳ Fetching data, please wait...")]
        )
    )

    # Process the heavy agent work in a background thread
    def process():
        try:
            reply = get_response_from_agent(user_text)
            if not reply:
                reply = "Sorry, I couldn't get a response."
        except Exception as e:
            reply = f"Error: {str(e)}"

        # Use push_message instead of reply_message (reply token is already used)
        line_bot_api.push_message(
            PushMessageRequest(
                to=event.source.user_id,
                messages=[TextMessage(text=reply)]
            )
        )

    threading.Thread(target=process).start()

def send_message_to_user(message: str, user_id: str = DEFAULT_USER_ID) -> None:
    # Keep this function fast-ish; any slow network issues are handled by calling it in background
    line_bot_api.push_message(
        PushMessageRequest(
            to=user_id,
            messages=[TextMessage(text=message)]
        )
    )


def _run_prompt_job(job_id: str, prompt: str, user_id: str) -> None:
    JOB_STATUS[job_id] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
    }

    try:
        response = get_response_from_agent(prompt)
        send_message_to_user(response, user_id=user_id)

        JOB_STATUS[job_id].update({
            "status": "done",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "response": response,
        })
    except Exception as e:
        JOB_STATUS[job_id].update({
            "status": "error",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })


@app.route("/prompt", methods=["GET"])
def get_prompt():
    return "Success", 200


@app.route("/prompt", methods=["POST"])
def post_prompt():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    # optional: allow caller to specify user_id; otherwise default
    user_id = data.get("user_id", DEFAULT_USER_ID)

    job_id = uuid.uuid4().hex
    JOB_STATUS[job_id] = {"status": "queued"}

    t = threading.Thread(target=_run_prompt_job, args=(job_id, prompt, user_id), daemon=True)
    t.start()

    # Respond immediately so Azure doesn't 504
    return jsonify({"status": "queued", "job_id": job_id}), 202


@app.get("/jobs/<job_id>")
def get_job(job_id: str):
    job = JOB_STATUS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)