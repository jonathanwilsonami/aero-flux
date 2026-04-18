#!/bin/bash

# Wait for network and apt to be fully ready
sleep 15

# System deps
apt-get update -y
apt-get install -y python3-pip git

# Pull code from GitHub
git clone https://github.com/jonathanwilsonami/aero-flux.git /home/ubuntu/aeroflux
chown -R ubuntu:ubuntu /home/ubuntu/aeroflux

# Install Python deps system-wide (no venv)
pip3 install --ignore-installed \
    dash dash-bootstrap-components plotly pandas numpy \
    pyarrow joblib xgboost boto3 gunicorn tensorflow

# systemd service
cat > /etc/systemd/system/aeroflux.service << 'EOF'
[Unit]
Description=AeroFlux - Digital Twin for Flight Data
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/aeroflux
ExecStart=/usr/local/bin/gunicorn \
    --workers 2 --timeout 180 --bind 0.0.0.0:8050 app:server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable aeroflux
systemctl start aeroflux