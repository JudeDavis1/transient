echo "[*] Stopping transient container"
docker stop transient

echo "[*] Removing transient container"
docker rm transient

echo "[*] Removing transient image"
docker rmi transient
