#!/bin/bash
# Reference boot script for a Debian Compute Engine VM.
# It is not installed automatically. Review it before use.
set -euo pipefail

apt-get update
apt-get install -y docker.io docker-compose-v2
systemctl enable --now docker

install -d -m 0750 /opt/astrohealth
# Place the repository at /opt/astrohealth and /etc/astrohealth.env (mode 600)
# before this script starts the containers.
if [[ -f /opt/astrohealth/docker-compose.prod.yml && -f /etc/astrohealth.env ]]; then
    docker compose -f /opt/astrohealth/docker-compose.prod.yml --env-file /etc/astrohealth.env up -d --build
fi