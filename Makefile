DOCKER := rebuild run

$(DOCKER):
	sh ./scripts/docker/$@.sh
