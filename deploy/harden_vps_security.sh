#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[vocaptest-harden] %s\n' "$*"
}

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Run this script as root." >&2
    exit 1
  fi
}

install_security_packages() {
  log "Installing firewall and ban tooling"
  apt-get update
  apt-get install -y --no-install-recommends ufw fail2ban
}

harden_ssh() {
  log "Hardening SSH: public-key root login remains enabled; password login is disabled"
  install -d -m 0755 /etc/ssh/sshd_config.d
  rm -f /etc/ssh/sshd_config.d/99-vocaptest-hardening.conf
  cat >/etc/ssh/sshd_config.d/00-vocaptest-hardening.conf <<'EOF'
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PermitRootLogin prohibit-password
PermitEmptyPasswords no
X11Forwarding no
EOF

  sshd -t
  systemctl reload ssh || systemctl reload sshd
}

configure_firewall() {
  log "Configuring UFW"
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow 22/tcp comment 'SSH'
  ufw allow 80/tcp comment 'HTTP'
  ufw allow 443/tcp comment 'HTTPS'
  ufw --force enable
}

configure_fail2ban() {
  log "Configuring fail2ban for sshd"
  install -d -m 0755 /etc/fail2ban/jail.d
  cat >/etc/fail2ban/jail.d/sshd.local <<'EOF'
[sshd]
enabled = true
port = ssh
maxretry = 5
findtime = 10m
bantime = 1h
EOF

  systemctl enable fail2ban
  systemctl restart fail2ban
}

print_status() {
  log "Status"
  sshd -T | grep -E '^(permitrootlogin|passwordauthentication|pubkeyauthentication|kbdinteractiveauthentication|x11forwarding) '
  ufw status verbose
  fail2ban-client status sshd || true
}

main() {
  require_root
  install_security_packages
  harden_ssh
  configure_firewall
  configure_fail2ban
  print_status
}

main "$@"
