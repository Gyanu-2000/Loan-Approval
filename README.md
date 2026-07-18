# Loan Approval Analyzer (Flask + scikit-learn)

## Setup

    pip install -r requirements.txt

## Run

    python server.py

Then open http://localhost:5000

## How it's wired together

- `model.py` loads `loan.csv`, imputes missing values, one-hot/label encodes
  categorical columns, trains a Logistic Regression model, and exposes:
  - `as_artifact()` — metadata for the UI (feature list, category options,
    numeric ranges, evaluation metrics)
  - `predict(state)` — scores a single application and returns a probability
    plus per-feature contributions
- `server.py` trains the model once at startup and serves two endpoints:
  - `GET /api/model` — returns `as_artifact()` as JSON
  - `POST /api/predict` — accepts a JSON form state and returns a prediction
- `templates/index.html` (originally `loan_approval_analyzer.html`) now
  fetches `/api/model` on load to build the form, and calls `/api/predict`
  each time a slider or dropdown changes, instead of using a hardcoded
  snapshot of the model baked into the page.
  > > > > > > > af00762 (Fix Flask template loading and API-backed UI)

=======

# Loan Approval Analyzer (Flask + scikit-learn)

## Setup

pip install -r requirements.txt

## Run

python server.py

Then open http://localhost:5000

## How it's wired together

- `model.py` loads `loan.csv`, imputes missing values, one-hot/label encodes
  categorical columns, trains a Logistic Regression model, and exposes:
  - `as_artifact()` — metadata for the UI (feature list, category options,
    numeric ranges, evaluation metrics)
  - `predict(state)` — scores a single application and returns a probability
    plus per-feature contributions
- `server.py` trains the model once at startup and serves two endpoints:
  - `GET /api/model` — returns `as_artifact()` as JSON
  - `POST /api/predict` — accepts a JSON form state and returns a prediction
- `templates/index.html` fetches `/api/model` on load to build the form, and calls
  `/api/predict` each time a slider or dropdown changes.

=======

# Loan Approval Analyzer (Flask + scikit-learn)

## Setup

    pip install -r requirements.txt

## Run

    python server.py

Then open http://localhost:5000

## How it's wired together

- `model.py` loads `loan.csv`, imputes missing values, one-hot/label encodes
  categorical columns, trains a Logistic Regression model, and exposes:
  - `as_artifact()` — metadata for the UI (feature list, category options,
    numeric ranges, evaluation metrics)
  - `predict(state)` — scores a single application and returns a probability
    plus per-feature contributions
- `server.py` trains the model once at startup and serves two endpoints:
  - `GET /api/model` — returns `as_artifact()` as JSON
  - `POST /api/predict` — accepts a JSON form state and returns a prediction
- `templates/index.html` (originally `loan_approval_analyzer.html`) now
  fetches `/api/model` on load to build the form, and calls `/api/predict`
  each time a slider or dropdown changes, instead of using a hardcoded
  snapshot of the model baked into the page.
  > > > > > > > af00762 (Fix Flask template loading and API-backed UI)
