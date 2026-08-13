# Deploy (infra/)

Deployment is fully CI/CD via GitHub Actions (no local AWS credentials needed).
The FastAPI app runs on AWS Lambda (container image) behind an HTTP API Gateway;
the Lambda streams the model from S3 at cold start. Scale-to-zero: idle cost ~$0.

Two layers:

- **infra/bootstrap** — one-time setup, run from AWS CloudShell. Creates the OIDC
  deploy role, the Terraform state backend (S3 + DynamoDB), the model bucket, and
  the budget alarm. These persist across deploy/destroy cycles.
- **infra/** — the app stack (ECR, Lambda, API Gateway). Deployed and destroyed by
  the GitHub Actions workflows using the OIDC role and remote state.

## 1. Bootstrap (once, in AWS CloudShell)

CloudShell (the browser terminal in the AWS console) already has your credentials.

```bash
# install terraform
curl -fsSLo terraform.zip https://releases.hashicorp.com/terraform/1.15.8/terraform_1.15.8_linux_amd64.zip
unzip -o terraform.zip && sudo mv terraform /usr/local/bin/

# get the code and apply the bootstrap
git clone https://github.com/KevDP/credit-risk-ml-pipeline.git
cd credit-risk-ml-pipeline
terraform -chdir=infra/bootstrap init
terraform -chdir=infra/bootstrap apply -var 'budget_email=you@example.com'
```

Upload the model (CloudShell **Actions > Upload file** > `models/model.joblib`), then:

```bash
aws s3 cp model.joblib "s3://$(terraform -chdir=infra/bootstrap output -raw model_bucket)/model.joblib"
```

## 2. Configure GitHub

Repo > Settings > Secrets and variables > Actions > **Variables**: add
`AWS_DEPLOY_ROLE_ARN` set to:

```bash
terraform -chdir=infra/bootstrap output -raw deploy_role_arn
```

## 3. Deploy

GitHub > Actions > **Deploy** > Run workflow. It assumes the OIDC role, creates
ECR, builds and pushes the image, and applies the Lambda + API Gateway. The
scoring URL is printed at the end of the run.

## 4. Test

```bash
curl -s -X POST "<predict_url>" -H "Content-Type: application/json" \
  -d '{"loan_amnt":10000,"annual_inc":60000,"dti":15.0,"open_acc":8,"pub_rec":0,"revol_bal":5000,"revol_util":30.0,"total_acc":20,"delinq_2yrs":0,"inq_last_6mths":1,"mort_acc":1,"pub_rec_bankruptcies":0,"fico_range_low":700,"fico_range_high":704,"term":" 36 months","emp_length":"5 years","issue_d":"Jan-2018","earliest_cr_line":"Jan-2005","home_ownership":"RENT","verification_status":"Verified","purpose":"credit_card","addr_state":"CA"}'
```

## 5. Tear down

GitHub > Actions > **Destroy** > Run workflow. Back to $0. The bootstrap resources
stay (they cost ~cents).

## Cost

Idle ~$0 (Lambda and API Gateway scale to zero). Persistent while deployed: the
ECR image (~$0.10/month) and the model in S3 (cents). Destroy removes the app
stack; the state bucket, model bucket, and budget in bootstrap persist.
