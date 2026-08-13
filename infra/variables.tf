variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region to deploy into."
}

variable "project_name" {
  type        = string
  default     = "credit-risk-scoring"
  description = "Name shared by the ECR repo, Lambda, and API. Must match bootstrap."
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
