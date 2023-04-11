echo "[*] Stopping and removing transient container"
docker stop transient 2>/dev/null || true

echo "[*] Removing transient image"
docker rm -f transient 2>/dev/null || true

echo "[*] Rebuilding transient image"
docker build docker -t transient

echo "[*] Running transient container"
docker run --name transient -d -it transient
