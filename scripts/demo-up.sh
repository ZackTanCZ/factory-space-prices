#!/bin/bash
# Deploy the ephemeral stack (ALB + ECS + networking) and wait for services to be ready.
# Run from the project root: bash scripts/demo-up.sh
# Takes ~10 minutes on first deploy.

set -e

REGION="${AWS_REGION:-ap-southeast-1}"
PERMANENT_STACK="fyp-permanent"
EPHEMERAL_STACK="fyp-ephemeral"
PROJECT_NAME="fyp"

echo "==> Querying default VPC..."
VPC_ID=$(aws ec2 describe-vpcs \
  --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' \
  --output text \
  --region "$REGION")

if [ -z "$VPC_ID" ] || [ "$VPC_ID" = "None" ]; then
  echo "ERROR: No default VPC found in region $REGION."
  exit 1
fi
echo "    VPC: $VPC_ID"

echo "==> Querying subnets in default VPC..."
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[*].SubnetId' \
  --output text \
  --region "$REGION" | tr '\t' ',')
echo "    Subnets: $SUBNET_IDS"

echo "==> Reading permanent stack outputs..."
get_output() {
  aws cloudformation describe-stacks \
    --stack-name "$PERMANENT_STACK" \
    --query "Stacks[0].Outputs[?OutputKey==\`$1\`].OutputValue" \
    --output text \
    --region "$REGION"
}

TASK_EXEC_ROLE=$(get_output TaskExecutionRoleArn)
BACKEND_ECR=$(get_output BackendECRUri)
FRONTEND_ECR=$(get_output FrontendECRUri)

if [ -z "$TASK_EXEC_ROLE" ] || [ "$TASK_EXEC_ROLE" = "None" ]; then
  echo "ERROR: Could not read permanent stack outputs."
  echo "       Deploy the permanent stack first: aws cloudformation deploy --template-file cfn/permanent.yml ..."
  exit 1
fi

echo "    TaskExecutionRole: $TASK_EXEC_ROLE"
echo "    BackendECR:        ${BACKEND_ECR}:latest"
echo "    FrontendECR:       ${FRONTEND_ECR}:latest"

echo ""
echo "==> Deploying ephemeral stack (this takes ~5 minutes)..."
aws cloudformation deploy \
  --template-file cfn/ephemeral.yml \
  --stack-name "$EPHEMERAL_STACK" \
  --parameter-overrides \
    ProjectName="$PROJECT_NAME" \
    TaskExecutionRoleArn="$TASK_EXEC_ROLE" \
    BackendImageUri="${BACKEND_ECR}:latest" \
    FrontendImageUri="${FRONTEND_ECR}:latest" \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNET_IDS" \
  --region "$REGION"

echo ""
echo "==> Reading ephemeral stack outputs..."
get_ephemeral_output() {
  aws cloudformation describe-stacks \
    --stack-name "$EPHEMERAL_STACK" \
    --query "Stacks[0].Outputs[?OutputKey==\`$1\`].OutputValue" \
    --output text \
    --region "$REGION"
}

CLUSTER=$(get_ephemeral_output ECSClusterName)
BACKEND_SVC=$(get_ephemeral_output BackendServiceName)
FRONTEND_SVC=$(get_ephemeral_output FrontendServiceName)
ALB_DNS=$(get_ephemeral_output ALBDNSName)

echo "    Cluster:  $CLUSTER"
echo "    Backend:  $BACKEND_SVC"
echo "    Frontend: $FRONTEND_SVC"

echo ""
echo "==> Waiting for ECS services to be stable (this takes ~5 minutes)..."
echo "    Backend is waiting for the XGBoost model to load before reporting healthy."
aws ecs wait services-stable \
  --cluster "$CLUSTER" \
  --services "$BACKEND_SVC" "$FRONTEND_SVC" \
  --region "$REGION"

echo ""
echo "========================================"
echo " Demo is live!"
echo "========================================"
echo "  Streamlit:   http://$ALB_DNS"
echo "  FastAPI docs: http://$ALB_DNS:8000/docs"
echo ""
echo "  Run 'bash scripts/demo-down.sh' after the demo to stop all charges."
echo "========================================"
