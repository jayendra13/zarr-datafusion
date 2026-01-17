variable "project_id" {
  description = "Existing GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "github_owner" {
  description = "GitHub repository owner"
  type        = string
  default     = "jayendra13"
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = "zarr-datafusion"
}

variable "github_connection_name" {
  description = "Name of the GitHub connection created in Cloud Build Console"
  type        = string
  default     = "github-connection"
}
