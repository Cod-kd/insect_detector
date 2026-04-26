# To run the project:
## BASH
1. Navigate to the project repository
1. Download dependencyes from Github: ` git clone --depth=1 https://github.com/tensorflow/tflite-micro.git ` 
1. Create standalone: ` cd tflite-micro && python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py   ../tflm_standalone `
-- 1. Setup files: ` ./setup_tflm.sh ./tflite-micro `
1. Remove repo: ` cd .. && rm -rf tflite-micro `
1. Build & Run with Docker: ` ./docker-build/_build.sh && ./docker-build/_run.sh `
