# Define targets and dependencies here


DOCKER := rebuild clean

default: help
.PHONY: help $(DOCKER)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "rebuild - build docker image"
	@echo ""
	@echo "clean - remove docker image and container"

$(DOCKER):
	sh ./scripts/docker/$@.sh
