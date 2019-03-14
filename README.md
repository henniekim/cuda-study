# cuda-study

## Introduction
This repository is for studying CUDA with the book "CUDA by example - an introduction to GPGPU programing". The code in this repository is originally from the book and I fixed a little bit in my favor. 

## How to compile
To compile *.cu file the command is like following
```shell
 nvcc mandelbrot.cu -o mandelbrot -lglut -lGL -lGLU
```
## How to execute
To execute the example
```shell
./mandelbrot
```

## Dependencies
1. OS : Ubuntu 16.04
2. CUDA Version : 10.1
1. GPU : NVIDIA GTX1070TI
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
