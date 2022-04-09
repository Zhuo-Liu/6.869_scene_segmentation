# 6.869_scene_segmentation
 2022 Spring 6.869 project

Steps to train:
1. Copy everything to Google Drive
2. Open /notebooks/run_scene_seg.ipynb on colab
3. Start training by running run_scene_seg.ipynb, change config by change the arg after --cfg.
4. To train on single GPU (only one GPU allowed on Colab), change the directories "root_dataset, list_train, list_val" under /config/ to Google Drive data directories. Also change "workers" to 2.