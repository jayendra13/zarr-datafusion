variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "github_owner" {
  description = "GitHub repository owner"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
}

variable "github_connection_name" {
  description = "Name of the manually-created GitHub connection in Cloud Build"
  type        = string
  default     = "github-connection"
}

variable "cloudbuild_sa_email" {
  description = "Cloud Build service account email"
  type        = string
}

variable "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  type        = string
}

variable "releases_bucket" {
  description = "GCS bucket for releases"
  type        = string
}

variable "cache_bucket" {
  description = "GCS bucket for build cache"
  type        = string
}
