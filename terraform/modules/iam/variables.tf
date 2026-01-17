variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "artifact_registry_repo" {
  description = "Artifact Registry repository ID"
  type        = string
}

variable "storage_bucket" {
  description = "Releases storage bucket name"
  type        = string
}

variable "cache_bucket" {
  description = "Cache storage bucket name"
  type        = string
}
