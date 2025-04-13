REVISION := $(shell git rev-parse --short HEAD)
    DOCKER_IMAGE_NAME = cr.yandex/crpu6b0he9vhd6gsbtje/fatsapi:$(REVISION)

build:
	docker build -q --platform linux/amd64 -t $(DOCKER_IMAGE_NAME) .

push:
	docker push $(DOCKER_IMAGE_NAME)