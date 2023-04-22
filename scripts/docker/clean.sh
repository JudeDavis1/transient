echo "[*] Stopping container"
docker stop -t 0 transient

echo "[*] Removing container"
docker rm transient

echo "[*] Removing image"
docker rmi transient

echo "[*] Removing volume"
docker volume rm ls -f $(docker volume ls -q)
