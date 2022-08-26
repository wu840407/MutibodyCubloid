# MultiBodyCuboid: Multi-body Segmentation and Motion Evaluation by Unsupervised Cuboid Shape Abstraction
## Introduction
MultiBodyCuboid is an multi-body motion segmentation architecture capable of handling an arbitrary number of disordered point sets.
## Dataset
Each dataset is organized in the following structure:

    <dataset-name>/
        ├ meta.json
        └ data/
            ├ 000000.npz
            ├ 000001.npz
            └ ...
After downloading the dataset, please set the paths in the corresponding yaml config files to the root of the dataset folder, i.e., `<dataset-name>/`.

### Articulated Objects
- Train+Val (`mbs-shapepart`): [Google Drive](https://drive.google.com/file/d/1aGTn-PYxLjnhj9UKlv4YFV3Mt1E3ftci/view?usp=sharing)
- Test (`mbs-sapien`): [Google Drive](https://drive.google.com/file/d/1HR2X0DjgXLwp8K5n2nsvfGTcDMSckX5Z/view?usp=sharing)
### Solid Objects
- Train+Val (`mbs-shapewhole`): [Google Drive](https://drive.google.com/file/d/1vAgavEzPJFG6lrwsl46ii1V5r3JM_zGR/view?usp=sharing)
- Test (`mbs-dynlab`): [Google Drive](https://drive.google.com/file/d/1sLOa-FfHzTslJ5MItKcAL5OQ7xr4_cju/view?usp=sharing)
