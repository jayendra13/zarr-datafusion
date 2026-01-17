# Releases bucket - stores release binaries
resource "google_storage_bucket" "releases" {
  project                     = var.project_id
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  lifecycle_rule {
    condition {
      age = 90
      matches_prefix = ["dev/"]
    }
    action {
      type = "Delete"
    }
  }

  versioning {
    enabled = true
  }
}

# Cache bucket - stores cargo build cache
resource "google_storage_bucket" "cache" {
  project                     = var.project_id
  name                        = "${var.project_id}-build-cache"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }
}
