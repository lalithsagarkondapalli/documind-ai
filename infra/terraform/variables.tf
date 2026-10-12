variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Name prefix for all resources."
  type        = string
  default     = "documind"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "single_nat_gateway" {
  description = "One shared NAT gateway (cheaper, not AZ-resilient). Set false for production."
  type        = bool
  default     = true
}

variable "eks_version" {
  type    = string
  default = "1.31"
}

variable "eks_public_access_cidrs" {
  description = "CIDRs allowed to reach the public Kubernetes API endpoint. Restrict this; do not leave 0.0.0.0/0 in production."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "node_instance_types" {
  type    = list(string)
  default = ["m6i.large"]
}

variable "node_min_size" {
  type    = number
  default = 3
}

variable "node_max_size" {
  type    = number
  default = 20
}

variable "node_desired_size" {
  type    = number
  default = 3
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.medium"
}

variable "db_allocated_storage" {
  type    = number
  default = 50
}

variable "db_multi_az" {
  type    = bool
  default = true
}

variable "db_deletion_protection" {
  description = "Protects the database from accidental `terraform destroy`."
  type        = bool
  default     = true
}

variable "redis_node_type" {
  type    = string
  default = "cache.t4g.small"
}

variable "github_repository" {
  description = "owner/repo allowed to assume the CI deploy role via GitHub OIDC (e.g. your-user/documind-ai)."
  type        = string
}
