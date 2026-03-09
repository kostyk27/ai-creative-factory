# Gunicorn config for Render.com
# https://docs.render.com/web-services#port-binding
import os

bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 1
timeout = 120
threads = 8
