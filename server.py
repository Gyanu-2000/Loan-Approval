"""
Flask app that connects loan.csv + the trained model (model.py) to the
loan_approval_analyzer.html frontend.

Run with:  python server.py
Then open: http://localhost:5000
"""

from flask import Flask, jsonify, render_template, request
import traceback
from jinja2 import FileSystemLoader

from model import LoanApprovalModel

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Ensure Flask can always find templates in case the working directory changes
app.template_folder = TEMPLATES_DIR
# Hard-set Jinja loader to the absolute templates directory (Render filesystem can differ)
app.jinja_loader = FileSystemLoader(TEMPLATES_DIR)

# Runtime guard: verify template dir contents are present.
# Render/Gunicorn sometimes bundles a different filesystem layout than local.
INDEX_TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "index.html")
if not os.path.exists(INDEX_TEMPLATE_PATH):
    app.logger.error(
        "Template not found at startup. Expected %s. Flask template_folder=%s",
        INDEX_TEMPLATE_PATH,
        app.template_folder,
    )



# Note: keep startup work lightweight.
# Avoid rendering templates at import/startup time.


# Train once on startup; reused for every request.
try:
    loan_model = LoanApprovalModel(os.path.join(BASE_DIR, "loan.csv"))
except Exception:
    app.logger.exception("Failed to initialize LoanApprovalModel")
    loan_model = None


LOG_FILE = os.path.join(BASE_DIR, "server_render_error.log")


def log_render_error(context: str, exc: BaseException) -> None:
    """Write full traceback to a file so Render/Gunicorn log truncation doesn't hide it."""
    tb = traceback.format_exc()
    if not tb or not tb.strip() or "NoneType: None" in tb:
        tb = repr(exc)

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n===== {context} =====\n")
            f.write(str(exc) + "\n")
            f.write(tb + "\n")
    except Exception:
        # Last resort: don't crash the request due to logging.
        pass


# Note: Do not render templates at import/startup time.
# Keep all template rendering inside request handlers (route functions)
# where Flask guarantees an application/request context.




@app.route("/", methods=["GET", "HEAD"])
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        # Return error details for debugging; also persist full traceback.
        app.logger.exception("Failed rendering index.html")
        log_render_error("render_template(index.html)", e)
        return (
            "Template rendering failed: " + str(e) + "\n\nSee server_render_error.log for traceback.",
            500,
        )


@app.route("/api/model")
def api_model():
    """Metadata the frontend needs: feature list, categories, ranges, metrics."""
    if loan_model is None:
        return jsonify({"error": "model failed to initialize"}), 500
    return jsonify(loan_model.as_artifact())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Body: { "Applicant_Income": 12000, "Gender": "Male", ... }"""
    if loan_model is None:
        return jsonify({"error": "model failed to initialize"}), 500
    state = request.get_json(force=True) or {}
    result = loan_model.predict(state)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
