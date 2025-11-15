import json
import logging

import azure.functions as func

from ..pdsqi_adapter import evaluate_pdsqi, save_to_table


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger for running PDSQI‑9 evaluation.

    Expects a JSON payload with the following fields:

    - summary (str): The summary text to evaluate.
    - notes (list[str]): A list of notes used to generate the summary.
    - target_specialty (str): The target clinician specialty.
    - model (str, optional): The name of the Azure OpenAI chat model to use. Defaults to "gpt-4o".

    Returns a JSON response containing the evaluation scores, the unique row identifier,
    and token usage information. The evaluated scores are also persisted to Azure Table
    Storage using the ``save_to_table`` helper from ``pdsqi_adapter``.
    """
    try:
        data = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON payload", status_code=400)

    # Extract required fields
    summary = data.get("summary")
    notes = data.get("notes") or []
    target_specialty = data.get("target_specialty")

    if not isinstance(summary, str) or not isinstance(notes, list) or not isinstance(target_specialty, str):
        return func.HttpResponse(
            "Request must include 'summary' (str), 'notes' (list of strings), and 'target_specialty' (str).",
            status_code=400,
        )

    # Optional model override
    model = data.get("model", "gpt-4o")

    try:
        # Perform evaluation
        parsed_scores, raw_output, usage = evaluate_pdsqi(summary, notes, target_specialty, model)

        # Persist to Azure Table Storage
        row_id = save_to_table(parsed_scores, raw_output)

        resp_body = {
            "id": row_id,
            "scores": parsed_scores,
            "usage": usage,
        }

        return func.HttpResponse(
            json.dumps(resp_body),
            mimetype="application/json",
            status_code=200,
        )

    except Exception as exc:
        logging.exception("Error running evaluation or persisting results")
        return func.HttpResponse(
            f"Internal server error: {str(exc)}",
            status_code=500,
        )