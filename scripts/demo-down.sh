#!/bin/bash
# Delete the ephemeral stack (ALB + ECS + networking) and stop all charges.
# Run from the project root: bash scripts/demo-down.sh
# Takes ~5 minutes.

set -e

REGION="${AWS_REGION:-ap-southeast-1}"
EPHEMERAL_STACK="fyp-ephemeral"

echo "==> Checking ephemeral stack exists..."
STATUS=$(aws cloudformation describe-stacks \
  --stack-name "$EPHEMERAL_STACK" \
  --query 'Stacks[0].StackStatus' \
  --output text \
  --region "$REGION" 2>/dev/null || echo "DOES_NOT_EXIST")

if [ "$STATUS" = "DOES_NOT_EXIST" ]; then
  echo "    Stack '$EPHEMERAL_STACK' does not exist — nothing to delete."
  exit 0
fi

echo "    Stack status: $STATUS"
echo ""
echo "==> Deleting ephemeral stack (this takes ~5 minutes)..."
aws cloudformation delete-stack \
  --stack-name "$EPHEMERAL_STACK" \
  --region "$REGION"

echo "    Waiting for deletion to complete..."
aws cloudformation wait stack-delete-complete \
  --stack-name "$EPHEMERAL_STACK" \
  --region "$REGION"

echo ""
echo "========================================"
echo " Ephemeral stack deleted."
echo " ALB, ECS cluster, and services are gone."
echo " All charges stopped."
echo ""
echo " ECR images and IAM roles are unaffected."
echo " Run 'bash scripts/demo-up.sh' before your next demo."
echo "========================================"
