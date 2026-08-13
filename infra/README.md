# Deploy (infra/)

The FastAPI app runs on AWS Lambda (container image) behind an HTTP API Gateway.
GitHub Actions builds and pushes the image (no local Docker); the Lambda streams
the model from S3 at cold start. Terraform runs locally. Scale-to-zero: idle cost is ~$0.
Tear down with `terraform destroy` to return to $0.

## Prerequisites

- AWS credentials configured locally (`aws configure`).
- Terraform >= 1.5.
- The GitHub OIDC provider must exist in the account. It usually already does; if
  not, add `-var 'create_github_oidc_provider=true'` to the first apply.

## Steps (run from the repository root)

1. Initialize and create the ECR repo, the CI role, and the model bucket:

   ```bash
   terraform -chdir=infra init
   terraform -chdir=infra apply \
     -target=aws_iam_role_policy.github_actions_ecr \
     -target=aws_s3_bucket.model
   ```

2. In GitHub → repo Settings → Secrets and variables → Actions → Variables, add
   `AWS_DEPLOY_ROLE_ARN` with the value of:

   ```bash
   terraform -chdir=infra output -raw github_actions_role_arn
   ```

3. Upload the trained model to S3:

   ```bash
   aws s3 cp models/model.joblib "s3://$(terraform -chdir=infra output -raw model_bucket)/model.joblib"
   ```

4. Run the **Build and push serving image** workflow (GitHub → Actions → Run
   workflow). It builds the image and pushes it to ECR.

5. Deploy the Lambda + API Gateway (optionally set an email for budget alerts):

   ```bash
   terraform -chdir=infra apply -var 'budget_email=you@example.com'
   ```

6. Test the live endpoint:

   ```bash
   curl -s -X POST "$(terraform -chdir=infra output -raw predict_url)" \
     -H "Content-Type: application/json" \
     -d '{"loan_amnt":10000,"annual_inc":60000,"dti":15.0,"open_acc":8,"pub_rec":0,"revol_bal":5000,"revol_util":30.0,"total_acc":20,"delinq_2yrs":0,"inq_last_6mths":1,"mort_acc":1,"pub_rec_bankruptcies":0,"fico_range_low":700,"fico_range_high":704,"term":" 36 months","emp_length":"5 years","issue_d":"Jan-2018","earliest_cr_line":"Jan-2005","home_ownership":"RENT","verification_status":"Verified","purpose":"credit_card","addr_state":"CA"}'
   ```

7. Tear down (back to $0):

   ```bash
   terraform -chdir=infra destroy
   ```

## Cost

Idle is ~$0 (Lambda and API Gateway scale to zero; free tier covers demo traffic).
Persistent cost is the ECR image (~$0.10/month) plus a few cents for the model in
S3, while they are stored. The budget alarm is a free tripwire.
