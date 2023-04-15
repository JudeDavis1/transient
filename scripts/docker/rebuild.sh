echo "[*] Stopping container"
docker stop transient 2>/dev/null || true

# echo "[*] Removing transient image"
# docker rm -f transient 2>/dev/null || true

echo "[*] Rebuilding image"
docker build -f ./docker/Dockerfile -t transient .

echo "[*] Running container"
docker run --name transient -d -it transient
