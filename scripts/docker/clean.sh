echo "[*] Stopping container"
docker compose stop -t 0 transient-base transient-test

echo "[*] Removing container"
docker compose rm base test

echo "[*] Removing image"
docker rmi -f $(docker images -q)

echo "[*] Removing volume"
docker volume rm ls -f $(docker volume ls -q)
