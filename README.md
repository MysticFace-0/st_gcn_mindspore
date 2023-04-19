# STGCN,  AGCN, PoseC3D  MindSpore
#### Dataloader:

​		FLAG2D(<u>**finished**</u>), FLAG3D(<u>**finished**</u>), FLAG2D_PoseConv3D(**<u>finished</u>**)

#### Model:

​		STGCN(**<u>finished</u>**), 2S-AGCN(**<u>finished</u>**), PoseConv3D(**<u>finished</u>**) 

#### Training file:

​		STGCN_FLAG2D(**<u>finished</u>**), STGCN_FLAG3D(**<u>finished</u>**),

​		2S-AGCN_FLAG2D(**<u>finished</u>**), 2S-AGCN_FLAG3D(**finished**),

​		PoseConv3D_FLAG2D(**<u>finished</u>**) 

#### pth2chpk:

​		STGCN_FLAG2D(**<u>finished, can maintain accuracy</u>**), 

​		STGCN_FLAG3D(**<u>finished, can maintain accuracy</u>**),

​		2S-AGCN_FLAG2D(**<u>finished, can maintain accuracy</u>**), 

​		2S-AGCN_FLAG3D(**finished, can maintain accuracy**),

​		PoseConv3D_FLAG2D(**<u>finished, can maintain accuracy</u>**) 

#### File introduction:

		- baseline: statement for model
		- chpk_rusume: where model parameters located
		- dataset: generate dataset of pose2d, pose3d
		- evaluation: evaluate function
		- logs: logger
		- model: stgcn, agcn, posec3d model
		- test: to evaluate model
		- utils: using for transform pth to chpk
		- test: using for develop test
		- train_2sagcn_flag2d.py: training file
		- train_2sagcn_flag3d.py:  training file
		- train_posec3d_flag2d.py:  training file
		- train_stgcn_flag2d.py:  training file
		- train_stgcn_flag3d.py: training file



Haonan Jiang(蒋浩楠) - Dalian University of Technology(大连理工大学)
