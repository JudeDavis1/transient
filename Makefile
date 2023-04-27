# Define targets and dependencies here


DOCKER := rebuild clean test

default: help
.PHONY: help $(DOCKER)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "rebuild - build docker image"
	@echo ""
	@echo "clean - remove docker image and container"
	@echo ""
	@echo "test - run test pipeline"

$(DOCKER):
	sh ./scripts/docker/$@.sh
