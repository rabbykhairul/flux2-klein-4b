variable "DOCKERHUB_REPO" {
  default = "zeroclue"
}

variable "DOCKERHUB_IMG" {
  default = "worker-comfyui-edit"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

variable "COMFYUI_VERSION" {
  default = "latest"
}

variable "BASE_IMAGE" {
  default = "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04"
}

variable "CUDA_VERSION_FOR_COMFY" {
  default = ""
}

variable "COMFY_CUSTOM_NODES" {
  default = "comfyui-image-saver"
}

variable "BUILD_VERSION" {
  default = ""
}

group "default" {
  targets = ["base", "qwen_image_edit_2511"]
}

target "base" {
  context = "."
  dockerfile = "Dockerfile"
  target = "base"
  platforms = ["linux/amd64"]
  args = {
    BASE_IMAGE = "${BASE_IMAGE}"
    COMFYUI_VERSION = "${COMFYUI_VERSION}"
    CUDA_VERSION_FOR_COMFY = "${CUDA_VERSION_FOR_COMFY}"
    COMFY_CUSTOM_NODES = "${COMFY_CUSTOM_NODES}"
    BUILD_VERSION = "${BUILD_VERSION}"
  }
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-base"]
}

target "qwen_image_edit_2511" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  args = {
    BASE_IMAGE = "${BASE_IMAGE}"
    COMFYUI_VERSION = "${COMFYUI_VERSION}"
    CUDA_VERSION_FOR_COMFY = "${CUDA_VERSION_FOR_COMFY}"
    COMFY_CUSTOM_NODES = "comfyui-image-saver"
    BUILD_VERSION = "${BUILD_VERSION}"
  }
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-qwen_image_edit_2511"]
  inherits = ["base"]
}
