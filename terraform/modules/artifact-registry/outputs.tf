output "repository_id" {
  description = "The repository ID"
  value       = google_artifact_registry_repository.docker.repository_id
}

output "repository_url" {
  description = "The repository URL for docker push/pull"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}
