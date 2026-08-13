output "api_endpoint" {
  value       = aws_apigatewayv2_api.this.api_endpoint
  description = "Base URL of the HTTP API."
}

output "predict_url" {
  value       = "${aws_apigatewayv2_api.this.api_endpoint}/predict"
  description = "Full URL of the scoring endpoint."
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.this.repository_url
  description = "ECR repo to push the Lambda image to."
}

output "github_actions_role_arn" {
  value       = aws_iam_role.github_actions.arn
  description = "Set this as the GitHub Actions repo variable AWS_DEPLOY_ROLE_ARN."
}

output "model_bucket" {
  value       = aws_s3_bucket.model.bucket
  description = "S3 bucket to upload the model artifact to."
}
