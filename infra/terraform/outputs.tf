output "cluster_name" {
  value = module.eks.cluster_name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.address
}

output "rds_master_secret_arn" {
  description = "Secrets Manager secret holding the generated RDS master credentials."
  value       = aws_db_instance.postgres.master_user_secret[0].secret_arn
}

output "app_secret_arn" {
  value = aws_secretsmanager_secret.app.arn
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "github_deploy_role_arn" {
  description = "Set this as the AWS_DEPLOY_ROLE_ARN repository variable used by .github/workflows/deploy.yml."
  value       = aws_iam_role.github_deploy.arn
}
