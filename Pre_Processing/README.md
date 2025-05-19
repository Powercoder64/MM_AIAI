Running instructions:

from the root direction (AIAI_CODES) run:
```
docker build -t pre-process -f .\Pre_Processing\docker_hub\Dockerfile .
```
After the images builds, run:
```
docker run --gpus all pre-process
```
Then enter the running container using either docker exec -it or docker desktop and change directory to Pre_Processing
Running 
```
python Pre_Processing.py messageid=123 requesttype=new filename=test.mp4
```
Will execute the code 
