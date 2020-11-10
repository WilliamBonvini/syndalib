# syndalib
Syndalib (Synthetic Data Library) is a python library that helps you create synthetic 2D point clouds for single/multi-model single/multi-class tasks.  
It makes you fix a set of hyperparameters for each class of models you are interested in generating.  
Models are saved in a .mat file format.  
<img src="media/imgs/scsm1.png" style="zoom:30%"> <img src="media/imgs/scsm2.png" style= "zoom:30%"> <img src="media/imgs/mcmm1.png" style="zoom:630%"> <img src="media/imgs/mcmm2.png" style="zoom:30%"> 
# setup


# usage
You can generate models of circles, lines and ellipses.  
You can define a vast set of parameters to specify the sampling space and the characteristics of your models (the hyperparameters change for each model, but each of them consists in a interval of values the hyperparameter can take).  
In this README you'll find a section for each class of models in which I'll dig deeper into the hyperparameters I provide.  
the generation process is straight-forward and it is shown in the following code snippet:

```python
# import the 2D point cloud module 
from syndalib import syn2d

# optionally you can specify the sampling space of both outliers and each class by defining a dictionary (options)
# and feeding it into the set_options() function.
# for reference, this example shows you the default options:
options = {
    "outliers": {
                "x_r": (-2.5, 2.5),
                "y_r": (-2.5, 2.5)
    },
     "circles": {
               "radius_r": (0.5, 1.5),
               "x_center_r": (-1.0, 1.0),
               "y_center_r": (-1.0, 1.0),
    },

    "lines": {
                "x_r": (-2.5, 2.5),
                "y_r": (-2.5, 2.5)
    },

    "ellipses": {
                "radius_r": (0.5, 1.5),
                "x_center_r": (-1, 1),
                "y_center_r": (-1, 1),
                "width_r": (0.1, 1),
                "height_r": (0.1, 1)
    },
}

syn2d.set_options(options)


# models generation
outliers_range = [0.1,0.2,0.3,0.4,0.5]
noise_range = [0.01]
syn2d.generate_data(ns=1024,
                    npps=256,
                    class_type="circles",
                    nm=2,
                    outliers_range=outliers_range,
                    noise_range=noise_range,
                    ds_name="example_dir",
                    is_train=False                 
                    )
```

# terminology and abbreviations
- sample: unordered set of points. a sample is made by outliers and inliers for each sampled model.
- model: instance of a class (i.e. line with a specific slope and intercept)
- npps: number of points per sample
- ns: number of samples within each .mat file
- nm: number of models to be generated within each sample of the dataset

# data folder
data are saved in a structured fashion.   
here I'll show you where the data generated in the previous code snippet will be saved:
```
./data
    |- circles
            |- nm_2
                 |- ds_name 
                         |- npps_256
                                  |- ns_1024
                                          |- test
                                                |- imgs
                                                |- circles_no_10_noise_0.01.mat
                                                |- circles_no_20_noise_0.01.mat
                                                |- circles_no_30_noise_0.01.mat
                                                |- circles_no_40_noise_0.01.mat
                                                |- circles_no_50_noise_0.01.mat
                                              
```
where ```imgs``` contains some images of the randomly sampled models. It has the following structure:
```
imgs
   |- circles_no_10_noise_0.01
                            |- *jpg files
   |- circles_no_20_noise_0.01
                            |- *jpg files 
   |- circles_no_30_noise_0.01
                            |- *jpg files
   |- circles_no_40_noise_0.01
                            |- *jpg files 
   |- circles_no_50_noise_0.01
                            |- *jpg files
```


# classes of models
## outliers
```class_type = "outliers" ```  
outliers are sampled from the 2D space $x_r \times y_r$,
where $x_r$ (x range) is the closed interval of values $x$ can take and 
$y_r$ (y range) is the closed interval of values $y$ can take.
```
"outliers": {
                "x_r": (-2.5, 2.5),
                "y_r": (-2.5, 2.5)
    },
```
## circles
```class_type = "circles"```  
circles are generated by uniformly sampling a center and a radius from the closed intervals specified by the user.  
the default values are:  
```
"circles": {
       "radius_r": (0.5, 1.5),
       "x_center_r": (-1.0, 1.0),
       "y_center_r": (-1.0, 1.0),
},
```

## lines
```class_type = "lines"```  
lines are generated by randomly sampling two points in the 2D space $x_r \times y_r$ (in order to define slope and intercept) 
and consequently sampling points belonging to such line.  
each point of the line is assured to belong to the 2D space $x_r \times y_r$.
```
"lines": {
            "x_r": (-2.5, 2.5),
            "y_r": (-2.5, 2.5)
},
```

## ellipses
```class_type = "ellipses"```  
ellipses are generated by uniformly sampling a center and a radius from the closed intervals specified by the user, and 
consequently apply a horizontal stretch to the sampled circle.  
the default values are:  
```
    "ellipses": {
                "radius_r": (0.5, 1.5),
                "x_center_r": (-1, 1),
                "y_center_r": (-1, 1),
                "width_r": (0.1, 1),
                "height_r": (0.1, 1)
    },
```


## conics
```class_type = "conics"```    
randomly samples models from the classes specified above.     
the hyperbola is not implemented yet.
