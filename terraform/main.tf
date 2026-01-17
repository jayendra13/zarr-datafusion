locals {
  repo_name = "zarr-datafusion"
}

module "artifact_registry" {
  source     = "./modules/artifact-registry"
  project_id = var.project_id
  region     = var.region
  repo_id    = local.repo_name
}

module "storage" {
  source      = "./modules/storage"
  project_id  = var.project_id
  region      = var.region
  bucket_name = "${var.project_id}-releases"
}

module "iam" {
  source                 = "./modules/iam"
  project_id             = var.project_id
  artifact_registry_repo = module.artifact_registry.repository_id
  storage_bucket         = module.storage.bucket_name
  cache_bucket           = module.storage.cache_bucket_name
  depends_on             = [module.artifact_registry, module.storage]
}

module "cloudbuild" {
  source                 = "./modules/cloudbuild"
  project_id             = var.project_id
  region                 = var.region
  github_owner           = var.github_owner
  github_repo            = var.github_repo
  github_connection_name = var.github_connection_name
  cloudbuild_sa_email    = module.iam.cloudbuild_sa_email
  artifact_registry_url  = module.artifact_registry.repository_url
  releases_bucket        = module.storage.bucket_name
  cache_bucket           = module.storage.cache_bucket_name
  depends_on             = [module.iam]
}
