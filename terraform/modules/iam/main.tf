# Custom service account for Cloud Build
resource "google_service_account" "cloudbuild" {
  project      = var.project_id
  account_id   = "cloudbuild-runner"
  display_name = "Cloud Build Runner"
  description  = "Service account for Cloud Build CI/CD pipelines"
}

# Artifact Registry Writer - push Docker images
resource "google_artifact_registry_repository_iam_member" "cloudbuild_writer" {
  project    = var.project_id
  location   = var.region
  repository = var.artifact_registry_repo
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cloudbuild.email}"
}

# Storage Admin for releases bucket
resource "google_storage_bucket_iam_member" "releases_admin" {
  bucket = var.storage_bucket
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloudbuild.email}"
}

# Storage Admin for cache bucket
resource "google_storage_bucket_iam_member" "cache_admin" {
  bucket = var.cache_bucket
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloudbuild.email}"
}

# Logs Writer - write build logs
resource "google_project_iam_member" "logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

# Cloud Build Builder - required for running builds
resource "google_project_iam_member" "cloudbuild_builds_builder" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.builder"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}
