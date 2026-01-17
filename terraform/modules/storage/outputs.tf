output "bucket_name" {
  description = "Releases bucket name"
  value       = google_storage_bucket.releases.name
}

output "cache_bucket_name" {
  description = "Cache bucket name"
  value       = google_storage_bucket.cache.name
}
