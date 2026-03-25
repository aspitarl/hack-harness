#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-app-ai-inc-dev-hack-directive-streamlit-alb}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-ai-inc-dev-hack-Team-ALB-eastus2-ioe}"
LOCATION="${LOCATION:-eastus2}"
PLAN_NAME="${PLAN_NAME:-asp-ai-inc-dev-hack-directive-streamlit-alb}"
ACR_NAME="${ACR_NAME:-acraiincdevhackalb}"
IMAGE_NAME="${IMAGE_NAME:-directive-investigation-streamlit}"
IMAGE_TAG="${IMAGE_TAG:-v1}"

echo "[1/7] Ensure resource group"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null

echo "[2/7] Ensure ACR"
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az acr create --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" --sku Basic >/dev/null
fi

ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)"
IMAGE_REF="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "[3/7] Build image in ACR: $IMAGE_REF"
az acr build --registry "$ACR_NAME" --image "${IMAGE_NAME}:${IMAGE_TAG}" . >/dev/null

echo "[4/7] Ensure App Service plan"
if ! az appservice plan show --name "$PLAN_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az appservice plan create --name "$PLAN_NAME" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" --is-linux --sku B1 >/dev/null
fi

echo "[5/7] Ensure Web App"
if ! az webapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az webapp create --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --plan "$PLAN_NAME" --container-image-name "$IMAGE_REF" >/dev/null
fi

echo "[6/7] Configure container + app settings"
az webapp config container set --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --container-image-name "$IMAGE_REF" >/dev/null

az webapp config appsettings set --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --settings \
  WEBSITES_PORT=8000 \
  SCM_DO_BUILD_DURING_DEPLOYMENT=false \
  CHAT_PROVIDER="${CHAT_PROVIDER:-foundry}" \
  AGENT_PROMPT_FILE="${AGENT_PROMPT_FILE:-agents/default.yaml}" \
  FOUNDRY_PROJECT_ENDPOINT="${FOUNDRY_PROJECT_ENDPOINT:-}" \
  FOUNDRY_API_KEY="${FOUNDRY_API_KEY:-}" \
  FOUNDRY_CHAT_DEPLOYMENT="${FOUNDRY_CHAT_DEPLOYMENT:-}" \
  FOUNDRY_API_VERSION="${FOUNDRY_API_VERSION:-2024-10-21}" \
  AZURE_AI_SEARCH_ENDPOINT="${AZURE_AI_SEARCH_ENDPOINT:-}" \
  AZURE_AI_SEARCH_API_KEY="${AZURE_AI_SEARCH_API_KEY:-}" \
  AZURE_AI_SEARCH_ORDERS_INDEX_NAME="${AZURE_AI_SEARCH_ORDERS_INDEX_NAME:-}" \
  AZURE_AI_SEARCH_PROCEDURES_INDEX_NAME="${AZURE_AI_SEARCH_PROCEDURES_INDEX_NAME:-}" \
  AZURE_AI_SEARCH_QUERY_TYPE="${AZURE_AI_SEARCH_QUERY_TYPE:-simple}" \
  AZURE_AI_SEARCH_IN_SCOPE="${AZURE_AI_SEARCH_IN_SCOPE:-true}" \
  AZURE_AI_SEARCH_STRICTNESS="${AZURE_AI_SEARCH_STRICTNESS:-3}" \
  AZURE_AI_SEARCH_TOP_N_DOCUMENTS="${AZURE_AI_SEARCH_TOP_N_DOCUMENTS:-5}" >/dev/null

echo "[7/7] Enable managed identity"
az webapp identity assign --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null

URL="https://${APP_NAME}.azurewebsites.net"
echo "Deployed: ${URL}"
