variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region to deploy into."
}

variable "project_name" {
  type        = string
  default     = "credit-risk-scoring"
  description = "Name shared by the ECR repo, Lambda, API, and budget."
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "ECR image tag the Lambda runs."
}

variable "memory_size" {
  type        = number
  default     = 2048
  description = "Lambda memory (MB). Enough for lightgbm + pandas."
}

variable "timeout" {
  type        = number
  default     = 30
  description = "Lambda timeout (seconds); covers cold start + inference."
}

variable "budget_limit" {
  type        = string
  default     = "5"
  description = "Monthly budget alarm threshold, in USD."
}

variable "budget_email" {
  type        = string
  default     = ""
  description = "Email for budget alerts. Leave empty to skip the notification."
}

variable "github_repo" {
  type        = string
  default     = "KevDP/credit-risk-ml-pipeline"
  description = "owner/repo allowed to assume the CI role via GitHub OIDC."
}

variable "create_github_oidc_provider" {
  type        = bool
  default     = false
  description = "Set true only if the GitHub OIDC provider does not already exist in the account."
}
