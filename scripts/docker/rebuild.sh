# echo "[*] Stopping container"
# docker stop -t 0.2 transient 2>/dev/null || true

# echo "[*] Removing image"
# docker rm -f transient 2>/dev/null || true

# echo "[*] Rebuilding image"
# docker build -f ./docker/Dockerfile -t transient .

# echo "[*] Running container"
# docker run --name transient -d -it transient

docker compose build
