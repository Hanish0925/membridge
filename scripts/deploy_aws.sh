#!/usr/bin/env bash
#
# Deploy MemBridge's demo to AWS: one Lambda behind a Function URL, one S3
# bucket holding both the static page and the ONNX model.
#
#   ./scripts/deploy_aws.sh
#
# Idempotent -- run it again to push new code. It creates, in this order:
#
#   s3://<bucket>                 static site + the MiniLM model
#   iam role membridge-lambda     basic execution + read on that one bucket
#   lambda membridge-api          the agent and its memory reads
#   a Function URL                public, unauthenticated, CORS-enabled
#
# Everything lands in us-east-1 because the CockroachDB cluster is in
# us-east-1. The agent makes several scoped vector queries per answer, so a
# cross-region hop would be paid repeatedly per request rather than once.
#
# Reads MEMBRIDGE_COCKROACH_DSN and GROQ_API_KEY from .env. Neither is baked
# into the package; both are set as Lambda environment variables.
#
# NOTE: this makes a bucket publicly readable, because a "functional demo URL"
# has to be reachable without credentials. The bucket holds the static page and
# a public model file; the DSN and the API key live only in Lambda's
# environment, never in S3 and never in the page.

set -euo pipefail

REGION="us-east-1"
FUNCTION="membridge-api"
ROLE="membridge-lambda"
RUNTIME="python3.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${REPO_ROOT}/build"

MODEL_SNAPSHOT="${HOME}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- preconditions ---------------------------------------------------------

command -v aws >/dev/null || die "aws CLI not found"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
  || die "AWS credentials are not configured; run 'aws configure'"

[ -f "${REPO_ROOT}/.env" ] || die ".env not found; it must hold MEMBRIDGE_COCKROACH_DSN"
set -a; . "${REPO_ROOT}/.env"; set +a
[ -n "${MEMBRIDGE_COCKROACH_DSN:-}" ] || die "MEMBRIDGE_COCKROACH_DSN is not set in .env"
[ -n "${GROQ_API_KEY:-}" ] || die "GROQ_API_KEY is not set in .env (get one at console.groq.com/keys)"

MODEL_ONNX="$(find "${MODEL_SNAPSHOT}" -name model.onnx 2>/dev/null | head -1)"
TOKENIZER="$(find "${MODEL_SNAPSHOT}" -name tokenizer.json 2>/dev/null | head -1)"
[ -n "${MODEL_ONNX}" ] || die "MiniLM ONNX model not in the HuggingFace cache; run the test suite once to fetch it"

BUCKET="membridge-demo-${ACCOUNT}"

say "account ${ACCOUNT}, region ${REGION}, bucket ${BUCKET}"

# --- the bucket ------------------------------------------------------------

if ! aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
  say "creating bucket"
  # us-east-1 is the one region that rejects a LocationConstraint.
  aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}"
  aws s3api put-public-access-block --bucket "${BUCKET}" \
    --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
  aws s3api put-bucket-policy --bucket "${BUCKET}" --policy "$(cat <<JSON
{"Version":"2012-10-17","Statement":[{"Sid":"PublicRead","Effect":"Allow",
"Principal":"*","Action":"s3:GetObject","Resource":"arn:aws:s3:::${BUCKET}/*"}]}
JSON
)"
  aws s3 website "s3://${BUCKET}" --index-document index.html
fi

say "uploading the MiniLM model (~90MB, once)"
aws s3 cp "${MODEL_ONNX}" "s3://${BUCKET}/model/model.onnx" --only-show-errors
aws s3 cp "${TOKENIZER}" "s3://${BUCKET}/model/tokenizer.json" --only-show-errors

# --- the deployment package ------------------------------------------------

say "building the Lambda package"
rm -rf "${BUILD}/pkg" && mkdir -p "${BUILD}/pkg"

# Linux wheels, not this machine's. Only what the handler actually reaches:
# mem0, sentence-transformers, torch and typer are all absent on purpose --
# migration runs from a laptop, retrieval runs here.
uv pip install --quiet \
  --python-platform x86_64-manylinux2014 --python-version 3.11 \
  --only-binary=:all: --target "${BUILD}/pkg" \
  onnxruntime numpy tokenizers "psycopg[binary]" pydantic

# onnxruntime declares sympy for symbolic shape inference, which inference never
# calls; huggingface_hub is only reached when MEMBRIDGE_ONNX_DIR is unset, and
# the handler always sets it after pulling the model from S3. Together they are
# ~44MB of a package that has to be re-uploaded on every code change.
rm -rf "${BUILD}/pkg"/{sympy,huggingface_hub,hf_xet,mpmath}
find "${BUILD}/pkg" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "${BUILD}/pkg" -name "*.dist-info" -type d -prune -exec rm -rf {} + 2>/dev/null || true

cp -r "${REPO_ROOT}/membridge" "${BUILD}/pkg/membridge"
find "${BUILD}/pkg/membridge" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

(cd "${BUILD}/pkg" && zip -qr "${BUILD}/lambda.zip" .)
printf '    package: %s unzipped, %s zipped\n' \
  "$(du -sh "${BUILD}/pkg" | cut -f1)" "$(du -h "${BUILD}/lambda.zip" | cut -f1)"

# --- the role --------------------------------------------------------------

if ! aws iam get-role --role-name "${ROLE}" >/dev/null 2>&1; then
  say "creating IAM role ${ROLE}"
  aws iam create-role --role-name "${ROLE}" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
      "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "${ROLE}" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  # Read on this bucket only, rather than AmazonS3ReadOnlyAccess.
  aws iam put-role-policy --role-name "${ROLE}" --policy-name membridge-model-read \
    --policy-document "$(cat <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"s3:GetObject",
"Resource":"arn:aws:s3:::${BUCKET}/model/*"}]}
JSON
)"
  say "waiting for the role to propagate to Lambda"
  sleep 12
fi
ROLE_ARN="$(aws iam get-role --role-name "${ROLE}" --query Role.Arn --output text)"

# --- the function ----------------------------------------------------------

ENV_VARS="Variables={MEMBRIDGE_COCKROACH_DSN=${MEMBRIDGE_COCKROACH_DSN},GROQ_API_KEY=${GROQ_API_KEY},MEMBRIDGE_ONNX_S3=${BUCKET}/model,MEMBRIDGE_LLM_PROVIDER=${MEMBRIDGE_LLM_PROVIDER:-groq}}"

if aws lambda get-function --function-name "${FUNCTION}" >/dev/null 2>&1; then
  say "updating ${FUNCTION}"
  aws lambda update-function-code --function-name "${FUNCTION}" \
    --zip-file "fileb://${BUILD}/lambda.zip" --query LastModified --output text
  aws lambda wait function-updated --function-name "${FUNCTION}"
  aws lambda update-function-configuration --function-name "${FUNCTION}" \
    --environment "${ENV_VARS}" --query LastModified --output text
else
  say "creating ${FUNCTION}"
  aws lambda create-function --function-name "${FUNCTION}" \
    --runtime "${RUNTIME}" --architectures x86_64 \
    --role "${ROLE_ARN}" \
    --handler membridge.serve.lambda_handler.handler \
    --zip-file "fileb://${BUILD}/lambda.zip" \
    --environment "${ENV_VARS}" \
    `# 1536MB is for CPU, not memory: Lambda scales cores with memory and the` \
    `# MiniLM forward pass is the slowest thing in a request.` \
    --memory-size 1536 \
    `# A cold start downloads 90MB from S3 and builds the ONNX session before` \
    `# it can answer; the agent may then make several LLM round trips.` \
    --timeout 60 \
    --query FunctionArn --output text
  aws lambda wait function-active --function-name "${FUNCTION}"
fi

if ! aws lambda get-function-url-config --function-name "${FUNCTION}" >/dev/null 2>&1; then
  say "creating the Function URL"
  aws lambda create-function-url-config --function-name "${FUNCTION}" \
    --auth-type NONE \
    --cors 'AllowOrigins=["*"],AllowMethods=["GET","POST"],AllowHeaders=["content-type"]' \
    --query FunctionUrl --output text
  aws lambda add-permission --function-name "${FUNCTION}" \
    --statement-id public-function-url --action lambda:InvokeFunctionUrl \
    --principal "*" --function-url-auth-type NONE >/dev/null
fi
API_URL="$(aws lambda get-function-url-config --function-name "${FUNCTION}" \
  --query FunctionUrl --output text)"

# --- the static site -------------------------------------------------------

say "publishing the site"
mkdir -p "${BUILD}/site"
sed "s|__API_URL__|${API_URL%/}|" "${REPO_ROOT}/web/index.html" > "${BUILD}/site/index.html"
aws s3 cp "${BUILD}/site/index.html" "s3://${BUCKET}/index.html" \
  --content-type "text/html; charset=utf-8" --cache-control "no-cache" --only-show-errors

SITE_URL="http://${BUCKET}.s3-website-${REGION}.amazonaws.com"

say "deployed"
printf '    demo : %s\n' "${SITE_URL}"
printf '    api  : %s\n' "${API_URL}"
printf '\n    check it: curl -s %shealth\n\n' "${API_URL}"
