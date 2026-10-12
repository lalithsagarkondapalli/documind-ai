terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state (create the bucket + lock table once, out-of-band, then uncomment):
  # backend "s3" {
  #   bucket         = "YOUR-tfstate-bucket"
  #   key            = "documind-ai/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "YOUR-tfstate-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "documind-ai"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
