# NOTE: Create the GitHub connection manually in GCP Console:
# Cloud Build > Repositories > Create host connection > GitHub
# Then link your repository through that connection.

locals {
  # Construct the repository ID directly
  repository_id = "projects/${var.project_id}/locations/${var.region}/connections/${var.github_connection_name}/repositories/${var.github_repo}"
}

# PR trigger - runs on pull requests to main
resource "google_cloudbuild_trigger" "pr" {
  project     = var.project_id
  location    = var.region
  name        = "pr-trigger"
  description = "Run CI on pull requests to main"

  repository_event_config {
    repository = local.repository_id
    pull_request {
      branch = "^main$"
    }
  }

  filename        = "cloudbuild/cloudbuild.yaml"
  service_account = "projects/${var.project_id}/serviceAccounts/${var.cloudbuild_sa_email}"

  substitutions = {
    _ARTIFACT_REGISTRY_URL = var.artifact_registry_url
    _RELEASES_BUCKET       = var.releases_bucket
    _CACHE_BUCKET          = var.cache_bucket
    _BUILD_TYPE            = "pr"
  }
}

# Main branch trigger - runs on push to main
resource "google_cloudbuild_trigger" "main" {
  project     = var.project_id
  location    = var.region
  name        = "main-trigger"
  description = "Run CI and push Docker image on merge to main"

  repository_event_config {
    repository = local.repository_id
    push {
      branch = "^main$"
    }
  }

  filename        = "cloudbuild/cloudbuild.yaml"
  service_account = "projects/${var.project_id}/serviceAccounts/${var.cloudbuild_sa_email}"

  substitutions = {
    _ARTIFACT_REGISTRY_URL = var.artifact_registry_url
    _RELEASES_BUCKET       = var.releases_bucket
    _CACHE_BUCKET          = var.cache_bucket
    _BUILD_TYPE            = "main"
  }
}

# Release trigger - runs on version tags
resource "google_cloudbuild_trigger" "release" {
  project     = var.project_id
  location    = var.region
  name        = "release-trigger"
  description = "Build and publish release on version tags"

  repository_event_config {
    repository = local.repository_id
    push {
      tag = "^v.*"
    }
  }

  filename        = "cloudbuild/cloudbuild-release.yaml"
  service_account = "projects/${var.project_id}/serviceAccounts/${var.cloudbuild_sa_email}"

  substitutions = {
    _ARTIFACT_REGISTRY_URL = var.artifact_registry_url
    _RELEASES_BUCKET       = var.releases_bucket
    _CACHE_BUCKET          = var.cache_bucket
  }
}
