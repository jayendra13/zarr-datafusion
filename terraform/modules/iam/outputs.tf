output "cloudbuild_sa_email" {
  description = "Cloud Build service account email"
  value       = google_service_account.cloudbuild.email
}

output "cloudbuild_sa_id" {
  description = "Cloud Build service account ID"
  value       = google_service_account.cloudbuild.id
}
