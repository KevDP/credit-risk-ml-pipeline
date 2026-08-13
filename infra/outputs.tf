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
  description = "ECR repo the image is pushed to."
}
