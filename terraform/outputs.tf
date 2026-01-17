output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = module.artifact_registry.repository_url
}

output "releases_bucket" {
  description = "GCS bucket for release binaries"
  value       = module.storage.bucket_name
}

output "cache_bucket" {
  description = "GCS bucket for build cache"
  value       = module.storage.cache_bucket_name
}

output "cloudbuild_sa_email" {
  description = "Cloud Build service account email"
  value       = module.iam.cloudbuild_sa_email
}
