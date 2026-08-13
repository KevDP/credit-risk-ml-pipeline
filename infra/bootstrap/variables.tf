variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region."
}

variable "project_name" {
  type        = string
  default     = "credit-risk-scoring"
  description = "Name shared by the deploy role, buckets, lock table, and budget."
}

variable "github_repo" {
  type        = string
  default     = "KevDP/credit-risk-ml-pipeline"
  description = "owner/repo allowed to assume the deploy role via GitHub OIDC."
}

variable "create_github_oidc_provider" {
  type        = bool
  default     = false
  description = "Set true only if the GitHub OIDC provider does not already exist in the account."
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
