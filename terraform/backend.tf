# Uncomment and configure after creating the state bucket:
# gcloud storage buckets create gs://zarr-datafusion-terraform-state \
#   --location=us-central1 --uniform-bucket-level-access

# terraform {
#   backend "gcs" {
#     bucket = "zarr-datafusion-terraform-state"
#     prefix = "terraform/state"
#   }
# }

# For initial setup, use local backend (default)
