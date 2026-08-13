output "deploy_role_arn" {
  value       = aws_iam_role.deploy.arn
  description = "Set this as the GitHub Actions repo variable AWS_DEPLOY_ROLE_ARN."
}

output "state_bucket" {
  value       = aws_s3_bucket.state.bucket
  description = "S3 bucket holding the Terraform remote state."
}

output "lock_table" {
  value       = aws_dynamodb_table.lock.name
  description = "DynamoDB table for Terraform state locking."
}

output "model_bucket" {
  value       = aws_s3_bucket.model.bucket
  description = "Upload models/model.joblib here (aws s3 cp) so the Lambda can load it."
}
