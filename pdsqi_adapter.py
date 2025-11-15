"""
pdsqi_adapter
=================

This module exposes helper functions for evaluating clinical summaries using the
PDSQI‑9 instrument and persisting the results to Azure Table Storage. It
wraps the evaluation logic provided by the ``evaluation_instruments`` package
with a simple API suitable for use in an Azure Function or other services.

The key entry points are:

``evaluate_pdsqi(summary, notes, target_specialty, model)``
    Builds the evaluation prompt, calls the Azure OpenAI chat completion API,
    parses the JSON scores, and returns a tuple of parsed scores, raw content,
    and token usage.

``save_to_table(evaluation_dict, raw_output, partition_key)``
    Persists the numeric scores (and optionally explanations) to an Azure Table
    with a generated RowKey. The table name and storage connection string are
    read from environment variables.

To use this module, ensure that the following environment variables are set:

``AZURE_OPENAI_KEY``
    API key for Azure OpenAI.

``AZURE_OPENAI_ENDPOINT``
    Endpoint URL for Azure OpenAI (e.g., "https://my-openai-resource.openai.azure.com").

``AZURE_STORAGE_CONNECTION_STRING``
    Connection string for the Azure Storage account used to store results.

``AZURE_TABLE_NAME`` (optional)
    Name of the Azure Table to store results. Defaults to ``pdsqi9results``.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Tuple

from azure.data.tables import TableServiceClient
from openai import AzureOpenAI

from evaluation_instruments.instruments.pdsqi_9.pdsqi_prompt import resolve_prompt

_client: AzureOpenAI | None = None


def _get_openai_client() -> AzureOpenAI:
    """Lazily instantiate and cache the Azure OpenAI client.

    Reads API credentials from the ``AZURE_OPENAI_KEY`` and
    ``AZURE_OPENAI_ENDPOINT`` environment variables. The API version is
    hard-coded to ``2024-05-01-preview`` which is compatible with recent
    versions of the Azure OpenAI Chat Completion API.

    Returns
    -------
    AzureOpenAI
        A ready-to-use client instance.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise EnvironmentError(
                "Environment variables AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set."
            )
        _client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-05-01-preview",
            azure_endpoint=endpoint,
        )
    return _client


def completion_fn(model: str, messages: List[Dict[str, Any]]) -> Any:
    """Call the Azure OpenAI chat completion API.

    Parameters
    ----------
    model : str
        The model identifier to use (e.g., ``gpt-4o`` or ``gpt-35-turbo``).
    messages : list
        The message payload as constructed by the PDSQI‑9 ``resolve_prompt``.

    Returns
    -------
    Any
        The raw response object from the OpenAI SDK.
    """
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response


def evaluate_pdsqi(
    summary: str,
    notes: List[str],
    target_specialty: str,
    model: str = "gpt-4o",
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Evaluate a clinical summary using the PDSQI‑9 instrument.

    This helper constructs the prompt using the provided inputs, delegates
    completion to Azure OpenAI, then parses the JSON from the response and
    returns both the parsed scores and the raw output for persistence.

    Parameters
    ----------
    summary : str
        The clinical summary text to evaluate.
    notes : list of str
        The list of clinical notes that were used to produce the summary.
    target_specialty : str
        The intended clinician specialty (e.g., ``"Internal Medicine"``).
    model : str, optional
        The Azure OpenAI model name to use, by default ``"gpt-4o"``.

    Returns
    -------
    Tuple[Dict[str, Any], str, Dict[str, Any]]
        A tuple containing:
        - ``parsed``: A dictionary of scores (and possibly explanations) keyed by rubric names.
        - ``raw_output``: The raw JSON string returned by the model (as contained in the first chat message).
        - ``usage``: A dictionary describing token usage from the API response.
    """
    # Build the message array using the instrument's prompt resolver
    messages = resolve_prompt(
        summary_to_evaluate=summary,
        notes=notes,
        target_specialty=target_specialty,
    )
    # Get the raw response
    response = completion_fn(model, messages)

    # Extract content and token usage
    content: str = response.choices[0].message.content
    # Azure OpenAI usage objects implement ``model_dump`` in 1.x of openai SDK
    if hasattr(response.usage, "model_dump"):
        usage = response.usage.model_dump()
    else:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    # Parse the JSON; errors will propagate for callers to handle
    parsed: Dict[str, Any] = json.loads(content)

    return parsed, content, usage


def save_to_table(
    evaluation_dict: Dict[str, Any],
    raw_output: str,
    partition_key: str = "PDSQI9",
) -> str:
    """Persist the evaluation results to Azure Table Storage.

    Each evaluation is stored in its own entity. The entity schema is
    flattened from the instrument's parsed output: for each rubric key, if
    the value is a dictionary with a ``score`` field (i.e., explanation and
    score), the score is stored; otherwise, the value itself is stored. The
    raw JSON output from the model is also stored for reference under the
    ``RawOutput`` property.

    Parameters
    ----------
    evaluation_dict : dict
        Parsed scores from the PDSQI‑9 evaluation.
    raw_output : str
        The raw JSON string produced by the model.
    partition_key : str, optional
        PartitionKey value for Azure Table entities, by default ``"PDSQI9"``.

    Returns
    -------
    str
        The generated RowKey for the stored entity.
    """
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise EnvironmentError(
            "AZURE_STORAGE_CONNECTION_STRING environment variable must be set."
        )
    table_name = os.environ.get("AZURE_TABLE_NAME", "pdsqi9results")
    # Create a TableServiceClient and get the table client
    service_client = TableServiceClient.from_connection_string(conn_str)
    table_client = service_client.get_table_client(table_name)
    # Ensure the table exists
    try:
        table_client.create_table()
    except Exception:
        # ignore if already exists
        pass

    row_id = str(uuid.uuid4())

    entity = {
        "PartitionKey": partition_key,
        "RowKey": row_id,
        "RawOutput": raw_output,
    }
    # Flatten evaluation dictionary into entity properties
    for key, value in evaluation_dict.items():
        # Normalize key: ensure no spaces or special chars for Table Storage
        norm_key = key.replace(" ", "_").replace("-", "_")
        if isinstance(value, dict) and "score" in value:
            entity[norm_key] = value.get("score")
        else:
            entity[norm_key] = value

    # Upsert the entity (create or replace)
    table_client.upsert_entity(entity)

    return row_id