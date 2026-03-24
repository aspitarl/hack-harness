"""Search Azure AI Search and print top documents.
Install Requirements

pip3 install -r quickpython/requirements.txt
Usage examples:
python testseach.py --service-name mysearch --index-name docs --query "azure"
python testseach.py --endpoint https://mysearch.search.windows.net --index-name docs --query "*" --key <admin-or-query-key>
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Query Azure AI Search and return the top documents. "
			"If --key is omitted, DefaultAzureCredential is used."
		)
	)

	endpoint_group = parser.add_mutually_exclusive_group(required=True)
	endpoint_group.add_argument(
		"--endpoint",
		help="Full search endpoint, for example https://<service>.search.windows.net",
	)
	endpoint_group.add_argument(
		"--service-name",
		help="Search service name used to build endpoint as https://<service>.search.windows.net",
	)

	parser.add_argument("--index-name", required=True, help="Azure AI Search index name")
	parser.add_argument("--query", required=True, help="Search text, for example '*' or 'contoso'")
	parser.add_argument("--key", default=None, help="Optional Azure AI Search query/admin key")
	parser.add_argument("--top", type=int, default=10, help="Number of documents to return (default: 10)")
	parser.add_argument("--filter", default=None, help="Optional OData filter")
	parser.add_argument("--select", default=None, help="Comma-separated fields to return")
	parser.add_argument("--order-by", default=None, help="OData order-by expression")
	parser.add_argument(
		"--include-total-count",
		action="store_true",
		help="Include total count in the response",
	)
	parser.add_argument(
		"--log-level",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
		default="INFO",
		help="Log verbosity",
	)

	return parser


def resolve_endpoint(endpoint: Optional[str], service_name: Optional[str]) -> str:
	if endpoint:
		return endpoint.rstrip("/")
	return f"https://{service_name}.search.windows.net"


def build_client(endpoint: str, index_name: str, key: Optional[str]) -> SearchClient:
	if key:
		credential = AzureKeyCredential(key)
		logging.info("Using API key authentication.")
	else:
		credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
		logging.info("No key provided. Using DefaultAzureCredential fallback.")

	return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)


def parse_select(select_arg: Optional[str]) -> Optional[Iterable[str]]:
	if not select_arg:
		return None
	fields = [field.strip() for field in select_arg.split(",") if field.strip()]
	return fields or None


def normalize_results(results: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
	normalized = []
	for item in results:
		normalized.append(dict(item))
	return normalized


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

	if args.top <= 0:
		parser.error("--top must be greater than 0")

	endpoint = resolve_endpoint(args.endpoint, args.service_name)

	# Optional env var fallback for key when argument is omitted.
	key = args.key or os.getenv("AZURE_SEARCH_API_KEY")

	try:
		client = build_client(endpoint=endpoint, index_name=args.index_name, key=key)

		select_fields = parse_select(args.select)
		response = client.search(
			search_text=args.query,
			top=args.top,
			filter=args.filter,
			select=select_fields,
			order_by=args.order_by,
			include_total_count=args.include_total_count,
		)

		documents = normalize_results(response)
		payload: Dict[str, Any] = {"count": len(documents), "documents": documents}

		if args.include_total_count:
			payload["total_count"] = response.get_count()

		print(json.dumps(payload, ensure_ascii=False, indent=2))
		return 0
	except HttpResponseError as exc:
		logging.error("Search request failed: %s", exc)
		return 1
	except Exception as exc:  # pylint: disable=broad-except
		logging.error("Unexpected failure: %s", exc)
		return 1


if __name__ == "__main__":
	sys.exit(main())
