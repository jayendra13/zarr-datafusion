output "pr_trigger_id" {
  description = "PR trigger ID"
  value       = google_cloudbuild_trigger.pr.trigger_id
}

output "main_trigger_id" {
  description = "Main branch trigger ID"
  value       = google_cloudbuild_trigger.main.trigger_id
}

output "release_trigger_id" {
  description = "Release trigger ID"
  value       = google_cloudbuild_trigger.release.trigger_id
}
