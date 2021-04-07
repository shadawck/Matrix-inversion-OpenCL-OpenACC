## OPENCL & OPENCACC

Pour compiler les deux solution (openACC et openCL), 
éxecuter les instructions suivantes à la racine du projet :
```shell
> mkdir build && cd build 
> cmake .. && make
```

Cela crée un dossier pour chaque projet avec chacun un éxecutable.
Donc pour openACC faites (à partir du dossier de build): 
```shell
> cd tp4_openacc
> ./pp_openacc <matrixDimension>
```
Pareil pour openCL :
```shell
> cd tp4_pp_opencl
> ./pp_opencl <matrixDimension>
```

Vous pouvez également compiler les projets séparement : 
Pour compiler seulement openACC, faite à la racine du projet : 
```shell
> cd tp4_openacc 
> mkdir build && cd build 
> cmake .. && make 
# Et puis executer le programme
> ./pp_openacc <matrixDimension>
```

Pour compiler seulement openCL, commenter la ligne indiquée dans le CmakeList de OpenCl et 
faites à la racine du projet:
```shell
> cd tp4_opencl
> mkdir build && cd build 
> cmake .. && make 
# Et puis executer le programme
> ./pp_opencl <matrixDimension>
```
