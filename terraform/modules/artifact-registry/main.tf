resource "google_artifact_registry_repository" "docker" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repo_id
  format        = "DOCKER"
  description   = "Docker images for zarr-datafusion"

  cleanup_policies {
    id     = "delete-old-dev-images"
    action = "DELETE"
    condition {
      tag_state  = "TAGGED"
      tag_prefixes = ["pr-", "dev-"]
      older_than = "604800s" # 7 days
    }
  }

  cleanup_policies {
    id     = "keep-release-images"
    action = "KEEP"
    condition {
      tag_state    = "TAGGED"
      tag_prefixes = ["v", "latest"]
    }
  }
}
