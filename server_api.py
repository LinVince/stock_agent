from flask import Blueprint, request, jsonify
from stock_agent import get_response_from_agent

agent_bp = Blueprint("agent", __name__)


@agent_bp.route("/prompt", methods=["GET"])
def get_prompt():
    prompt = request.args.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt' query parameter"}), 400
    response = get_response_from_agent(prompt)
    return jsonify({"prompt": prompt, "response": response})


@agent_bp.route("/prompt", methods=["POST"])
def post_prompt():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400
    prompt = data["prompt"]
    response = get_response_from_agent(prompt)
    return jsonify({"prompt": prompt, "response": response})