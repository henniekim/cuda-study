# cuda-study

## Introduction
This repository is for studying CUDA with the book "CUDA by example - an introduction to GPGPU programing". The code in this repository is originally from the book and I fixed a little bit in my favor. 

## 1. Contents
### [1.1 cuda-by-examples](/cuda-by-examples)
- ch04   
   - [vector addition](/cuda-by-examples/vector_addition.cu)
   - [julia set](cuda-by-examples/julia_set_examples.cu)  
   - [mandelbrot](/cuda-by-examples/vector_addition.cu)  
- ch05
  - [waving](/cuda-by-examples/wave_using_thread.cu)
  - [shared memory](/cuda-by-examples/shared_memory.cu)  


### [1.2 cuda-image-processing](/cuda-image-processing)
- intensity transformation
  - [negative](/cuda-image-processing/image_negative.cu)
  -  

## 2. How to compile
To compile *.cu file the command is like following
```shell
 nvcc mandelbrot.cu -o mandelbrot -lglut -lGL -lGLU // 
```
## 3. How to execute
To execute the example
```shell
./mandelbrot.cubin
```

## 4. Dependencies
1. OS : Ubuntu 16.04
2. CUDA Version : 10.1
3. GPU : NVIDIA GTX1070TI
4. FreeGlut 
## For VSCode Users
for vscode the tasks.json file will be like this

```json

{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "save and compile for CUDA",
            "type": "shell",
            "command": "nvcc",
            "args": [
                "${fileDirname}/${fileBasenameNoExtension}.cu",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}",
                "-lglut",
                "-lGL",
                "-lGLU"
            
            ],
            "group" : "build",

            "problemMatcher": {
                "fileLocation": [
                    "relative",
                    "${workspaceRoot}"
                ],
                "pattern": {
                    // The regular expression. 
                   //Example to match: helloWorld.c:5:3: warning: implicit declaration of function 'prinft'
                    "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning error):\\s+(.*)$",
                    "file": 1,
                    "line": 2,
                    "column": 3,
                    "severity": 4,
                    "message": 5
                }
            }
        
        },

        // execute binary ! (ubuntu 16.04)
        {
            "label": "execute",
            "type": "shell",
            "command": "${fileDirname}/${fileBasenameNoExtension}",
            "group" : "test"

        }
    ]
}
```
