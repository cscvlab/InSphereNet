# InSphereNet: a Concise Representation andClassification Method for 3D Object

## Abstract
In this paper, we present an InSphereNet method for the problem of 3D object classification. Unlike previous methods that use points, voxels, or multi-view images as inputs of deep neural network (DNN), the proposed method constructs a class of more representative features named infilling spheres from signed distance field (SDF). Because of the admirable spatial representation of infilling spheres, we can not only utilize very fewer number of spheres to accomplish classification task, but also design a lightweight InSphereNet with less layers and parameters than previous methods. Experiments on ModelNet40 show that the proposed method leads to superior performance than PointNet in accuracy. In particular, if there are only a few dozen sphere inputs or about 100000 DNN parameters, the accuracy of our method remains at a very high level.<br>
## Our Approach
Firstly, we voxelize the 3Dmodel with a high resolution of 512 x 512 x 512. Secondly, the SDF value of each voxel within an external sphere is computed. Thirdly, a number of voxels with larger SDF values are selected according to three criteria. Finally, positions and radii of selected infilling spheres are fed into the classification network. The detailed workflow is illustrated in Fig. 2.<br>
![image1](https://github.com/cscvlab/InSphereNet/blob/master/flow.JPG)
![image2](https://github.com/cscvlab/InSphereNet/blob/master/air.JPG)
![image3](https://github.com/cscvlab/InSphereNet/blob/master/network.JPG)
![image3](https://github.com/cscvlab/InSphereNet/blob/master/experiment.JPG)
